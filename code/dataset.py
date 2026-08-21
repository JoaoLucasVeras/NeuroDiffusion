from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
from natsort import natsorted
from glob import glob
import pickle

from transformers import AutoProcessor
def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img



#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore

class eeg_pretrain_dataset(Dataset):
    def __init__(self, path='../dreamdiffusion/datasets/mne_data/', roi='VC', patch_size=16, transform=identity, aug_times=2, 
                num_sub_limit=None, include_kam=False, include_hcp=True):
        super(eeg_pretrain_dataset, self).__init__()
        data = []
        images = []
        self.input_paths = [str(f) for f in sorted(Path(path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]

        assert len(self.input_paths) != 0, 'No data found'
        self.data_len  = 512
        self.data_chan = 128

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, index):
        data_path = self.input_paths[index]

        data = np.load(data_path)

        if data.shape[-1] > self.data_len:
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len)+1)

            data = data[:, idx: idx+self.data_len]
        else:
            x = np.linspace(0, 1, data.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, data)
            data = f(x2)
        ret = np.zeros((self.data_chan, self.data_len))
        if (self.data_chan > data.shape[-2]):
            for i in range((self.data_chan//data.shape[-2])):

                ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data
            if self.data_chan % data.shape[-2] != 0:

                ret[ -(self.data_chan%data.shape[-2]):, :] = data[: (self.data_chan%data.shape[-2]), :]
        elif(self.data_chan < data.shape[-2]):
            idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan)+1)
            ret = data[idx2: idx2+self.data_chan, :]
        # print(ret.shape)
        elif(self.data_chan == data.shape[-2]):
            ret = data
        ret = ret/10 # reduce an order
        # torch.tensor()
        ret = torch.from_numpy(ret).float()
        return {'eeg': ret } #,



def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                break
    return img_label, naive_label

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs


def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]

EEG_EXTENSIONS = [
    '.mat'
]


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in EEG_EXTENSIONS)


def make_dataset(dir):

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, topdown=False)):#
        for fname in fnames:
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

from PIL import Image
import numpy as np
 


def resolve_imagenet_dir(path):
    """Tolerate the double-nested imageNet_images/imageNet_images layout."""
    if path is None:
        return None
    nested = os.path.join(path, 'imageNet_images')
    if os.path.isdir(nested) and any(
            d.startswith('n') for d in os.listdir(nested)
            if os.path.isdir(os.path.join(nested, d))):
        return nested
    return path


class EEGDataset(Dataset):

    # Constructor
    def __init__(self, eeg_signals_path, imagenet_path, image_transform=identity, subject=0,
                 strict_images=True, missing_tolerance=0.0, load_images=True,
                 validate_scope=None):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)

        # Keep the ORIGINAL index of every retained trial. Splits are written against
        # the full trial list, so subject filtering must not silently renumber them.
        all_data = loaded['dataset']
        if subject != 0:
            kept = [i for i, e in enumerate(all_data) if e['subject'] == subject]
        else:
            kept = list(range(len(all_data)))
        self.data = [all_data[i] for i in kept]
        self.orig_to_pos = {orig: pos for pos, orig in enumerate(kept)}

        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = resolve_imagenet_dir(imagenet_path)
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        self.size = len(self.data)
        self.load_images = load_images

        # Stage 1 (masked EEG pre-training) is self-supervised and never touches the
        # stimulus. Skipping image IO there avoids a hard dependency on the image set
        # and removes a CLIP preprocess from every sample.
        if not self.load_images:
            self.processor = None
            self.missing = []
            print("[EEGDataset] image loading disabled (EEG-only mode), %d trials" % self.size)
            return

        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

        if self.imagenet is None:
            raise ValueError("No ImageNet path provided to EEGDataset. Training requires real images.")
        if not os.path.isdir(self.imagenet):
            raise FileNotFoundError("ImageNet directory not found at %s." % self.imagenet)

        # Verify the stimulus files exist UP FRONT. A missing file used to fall back to a
        # black square, which turns the diffusion target into a constant and makes the whole
        # run meaningless while the loss still looks like it is converging. Fail loudly instead.
        #
        # validate_scope restricts the check to the trials the splits actually reach. A split
        # built with --require_images has already excluded unavailable stimuli, so checking
        # the whole dataset would reject a perfectly clean configuration.
        if validate_scope is not None:
            scope = [self.data[self.orig_to_pos[i]] for i in validate_scope
                     if i in self.orig_to_pos]
        else:
            scope = self.data
        wanted = sorted({e['image'] for e in scope})
        self.missing = [s for s in wanted if not os.path.exists(self._image_path(s))]
        frac = len(self.missing) / max(1, len(wanted))
        print("[EEGDataset] %d/%d stimulus images found under %s"
              % (len(wanted) - len(self.missing), len(wanted), self.imagenet))
        if self.missing and strict_images and frac > missing_tolerance:
            preview = ', '.join(self.missing[:5])
            raise FileNotFoundError(
                "%d/%d stimulus images are missing (%.1f%% > tolerance %.1f%%).\n"
                "  Missing e.g.: %s\n"
                "  Training against absent images silently substitutes a blank target and "
                "produces a model that has learned nothing. Fetch the stimulus set, or pass "
                "strict_images=False if you have deliberately accepted the loss."
                % (len(self.missing), len(wanted), frac * 100, missing_tolerance * 100, preview)
            )
        if self.missing:
            print("[EEGDataset] WARNING: %d missing images will use a blank target." % len(self.missing))

    def _image_path(self, stem):
        return os.path.join(self.imagenet or '', stem.split('_')[0], stem + '.JPEG')

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):

        eeg = self.data[i]["eeg"].float().t()

        # Temporal jitter augmentation. The window is expressed as a fraction of the
        # recording so it applies to the 125-sample Shimizu trials as well as the
        # longer ImageNet-EEG ones (the old absolute >=460 gate never fired here).
        n_t = eeg.shape[0]
        if self.image_transform is not identity and n_t >= 16:
            max_shift = max(1, int(round(n_t * 0.08)))
            shift = np.random.randint(-max_shift, max_shift + 1)
            keep = n_t - max_shift
            start_idx = int(np.clip(max_shift // 2 + shift, 0, n_t - keep))
            eeg = eeg[start_idx:start_idx + keep, :]

        eeg = np.array(eeg.transpose(0, 1))
        # Resample the time axis to the length the encoder expects.
        if eeg.shape[-1] != self.data_len:
            x = np.linspace(0, 1, eeg.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, eeg)
            eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()

        label = torch.tensor(self.data[i]["label"]).long()

        if not self.load_images:
            return {'eeg': eeg, 'label': label, 'image': 0, 'image_raw': 0}

        image_name = self.data[i]["image"]
        image_path = self._image_path(image_name)
        if os.path.exists(image_path):
            image_raw_pil = Image.open(image_path).convert('RGB')
        else:
            image_raw_pil = Image.new('RGB', (512, 512), (0, 0, 0))

        image = np.array(image_raw_pil).astype(np.float32) / 255.0
        image_raw = self.processor(images=image_raw_pil, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}


class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=0):
        self.dataset = dataset
        loaded = torch.load(split_path)

        raw_idx = loaded["splits"][split_num][split_name]
        # Split indices address the FULL trial list. Map them onto this dataset's
        # positions so subject filtering cannot silently point at the wrong trials.
        mapped = [dataset.orig_to_pos[i] for i in raw_idx if i in dataset.orig_to_pos]
        self.split_idx = [i for i in mapped if dataset.data[i]["eeg"].size(1) >= 120]

        dropped = len(raw_idx) - len(self.split_idx)
        if dropped:
            print("[Splitter:%s] %d/%d split indices dropped (subject filter or short recording)"
                  % (split_name, dropped, len(raw_idx)))
        if not self.split_idx:
            raise ValueError(
                "Split '%s' is empty after mapping. The splits file was probably built for a "
                "different dataset than the one that was loaded." % split_name
            )

        self.size = len(self.split_idx)
        self.num_voxels = dataset.num_voxels
        self.data_len = dataset.data_len
        self.protocol = loaded.get("description", "unspecified")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


def create_EEG_dataset(eeg_signals_path='../datasets/imagination_5_95_std.pth',
            splits_path='../datasets/imagination_5_95_std_splits_subject.pth',
            imagenet_path=None,
            image_transform=identity, subject=0,
            strict_images=True, missing_tolerance=0.0, load_images=True):

    # Load the splits first so the stimulus check can be scoped to the trials that
    # training will actually touch, rather than to every trial in the file.
    validate_scope = None
    if load_images and splits_path and os.path.exists(splits_path):
        sp = torch.load(splits_path, map_location='cpu')['splits'][0]
        validate_scope = sorted(set(sp['train']) | set(sp['test']))

    if isinstance(image_transform, list):
        dataset_train = EEGDataset(eeg_signals_path, imagenet_path, image_transform[0], subject,
                                   strict_images, missing_tolerance, load_images, validate_scope)
        dataset_test = EEGDataset(eeg_signals_path, imagenet_path, image_transform[1], subject,
                                  strict_images, missing_tolerance, load_images, validate_scope)
    else:
        dataset_train = EEGDataset(eeg_signals_path, imagenet_path, image_transform, subject,
                                   strict_images, missing_tolerance, load_images, validate_scope)
        dataset_test = EEGDataset(eeg_signals_path, imagenet_path, image_transform, subject,
                                  strict_images, missing_tolerance, load_images, validate_scope)
    split_train = Splitter(dataset_train, split_path=splits_path, split_num=0, split_name='train', subject=subject)
    split_test = Splitter(dataset_test, split_path=splits_path, split_num=0, split_name='test', subject=subject)
    print("[create_EEG_dataset] protocol: %s" % split_train.protocol)
    print("[create_EEG_dataset] train=%d test=%d" % (len(split_train), len(split_test)))
    return (split_train, split_test)


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img



def normalize2(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img



def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')


if __name__ == '__main__':
    import scipy.io as scio
    import copy
    import shutil


