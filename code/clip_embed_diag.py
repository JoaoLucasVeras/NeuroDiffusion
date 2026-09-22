"""Are CLIP ViT-L/14 image embeddings of different stimuli nearly parallel?
If mean pairwise cosine is ~1, a pointwise `1 - cos` alignment loss is degenerate."""
import glob, random, torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection
random.seed(0)
paths = sorted(glob.glob('/home/015555345/NeuroDiffusion/datasets/imageNet_images/*/*.JPEG'))
paths = random.sample(paths, 48)
proc = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
with torch.no_grad():
    e = m(**proc(images=[Image.open(p).convert('RGB') for p in paths], return_tensors='pt')).image_embeds
print('embed shape', tuple(e.shape), 'norm mean %.2f' % e.norm(dim=-1).mean())
en = torch.nn.functional.normalize(e, dim=-1)
cos = en @ en.T
off = cos[~torch.eye(len(e), dtype=bool)]
print('pairwise cosine: mean %.4f  min %.4f  max %.4f' % (off.mean(), off.min(), off.max()))
mean_vec = torch.nn.functional.normalize(e.mean(0, keepdim=True), dim=-1)
print('cosine to mean embedding: mean %.4f  -> 1-cos = %.4f' % ((en @ mean_vec.T).mean(), 1 - (en @ mean_vec.T).mean()))
top = e.abs().mean(0).topk(3)
print('largest |dims| (mean over images):', [(int(i), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
