"""
CLIP-based evaluation for EEG-to-image reconstruction, plus the baselines that make
the numbers interpretable.

A raw CLIP similarity is close to meaningless on its own -- two arbitrary natural
images already score ~0.5. What tells you whether the model extracted real
information from the EEG is the comparison against baselines that have no EEG
information in them:

  noise      generated-vs-noise images. Floor for "did anything render".
  shuffled   your own generations, paired against the WRONG ground truth. This is
             the critical one: it holds image quality constant and removes only the
             EEG-image correspondence. If your score does not beat shuffled, the
             model is producing plausible pictures unrelated to the brain signal.
  chance     1/N for the N-way classification metrics.

Everything here loads openai/clip-vit-large-patch14, the same checkpoint dataset.py
already pulls, so it resolves from the HF cache on an air-gapped compute node.
"""
import numpy as np
import torch
import torch.nn.functional as F

CLIP_MODEL = "openai/clip-vit-large-patch14"

# WordNet lemmas for the 40 ImageNet synsets in this stimulus set, used to build
# text prompts for zero-shot classification.
SYNSET_NAMES = {
    "n02106662": "German shepherd dog",
    "n02124075": "Egyptian cat",
    "n02281787": "lycaenid butterfly",
    "n02389026": "sorrel horse",
    "n02492035": "capuchin monkey",
    "n02504458": "African elephant",
    "n02510455": "giant panda",
    "n02607072": "anemone fish",
    "n02690373": "airliner",
    "n02906734": "broom",
    "n02951358": "canoe",
    "n02992529": "cellular telephone",
    "n03063599": "coffee mug",
    "n03100240": "convertible car",
    "n03180011": "desktop computer",
    "n03197337": "digital watch",
    "n03272010": "electric guitar",
    "n03272562": "electric locomotive",
    "n03297495": "espresso maker",
    "n03376595": "folding chair",
    "n03445777": "golf ball",
    "n03452741": "grand piano",
    "n03584829": "clothes iron",
    "n03590841": "jack-o'-lantern",
    "n03709823": "mailbag",
    "n03773504": "missile",
    "n03775071": "mitten",
    "n03792782": "mountain bike",
    "n03792972": "mountain tent",
    "n03877472": "pajamas",
    "n03888257": "parachute",
    "n03982430": "pool table",
    "n04044716": "radio telescope",
    "n04069434": "reflex camera",
    "n04086273": "revolver",
    "n04120489": "running shoe",
    "n07753592": "banana",
    "n07873807": "pizza",
    "n11939491": "daisy",
    "n13054560": "bolete mushroom",
}


def synset_to_name(synset):
    return SYNSET_NAMES.get(synset, synset)


class CLIPScorer:
    """Wraps CLIP for image-image similarity and zero-shot classification."""

    def __init__(self, device=None, model_name=CLIP_MODEL, batch_size=32):
        from transformers import CLIPModel, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    @torch.no_grad()
    def embed_images(self, imgs):
        """imgs: (n, h, w, 3) uint8 array. Returns L2-normalised (n, d) embeddings."""
        imgs = np.asarray(imgs)
        if imgs.shape[-1] != 3:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        out = []
        for i in range(0, len(imgs), self.batch_size):
            chunk = [im.astype(np.uint8) for im in imgs[i:i + self.batch_size]]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            out.append(F.normalize(feats, dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1).cpu()

    def image_similarity(self, pred_imgs, gt_imgs):
        """Per-pair cosine similarity between generated and ground-truth images."""
        a = self.embed_images(pred_imgs)
        b = self.embed_images(gt_imgs)
        return (a * b).sum(-1).numpy()

    def zeroshot_classify(self, pred_imgs, class_synsets, prompt="a photo of a {}"):
        """Return (n, n_classes) softmax probabilities over the class prompts."""
        texts = [prompt.format(synset_to_name(s)) for s in class_synsets]
        t = self.embed_texts(texts)
        v = self.embed_images(pred_imgs)
        logits = 100.0 * v @ t.T
        return logits.softmax(-1).numpy()

    def topk_accuracy(self, pred_imgs, true_labels, class_synsets, k=1):
        probs = self.zeroshot_classify(pred_imgs, class_synsets)
        topk = np.argsort(-probs, axis=1)[:, :k]
        hits = [int(t in row) for t, row in zip(true_labels, topk)]
        return float(np.mean(hits)), probs

    def n_way_accuracy(self, probs, true_labels, n_way=2, n_trials=100, seed=2022):
        """Restrict each decision to the true class plus n_way-1 random distractors."""
        rng = np.random.RandomState(seed)
        n_classes = probs.shape[1]
        scores = []
        for p, t in zip(probs, true_labels):
            wins = 0
            others = [c for c in range(n_classes) if c != t]
            for _ in range(n_trials):
                picks = rng.choice(others, n_way - 1, replace=False)
                if p[t] > max(p[c] for c in picks):
                    wins += 1
            scores.append(wins / n_trials)
        return float(np.mean(scores))


def make_noise_baseline(n, h=512, w=512, seed=2022):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (n, h, w, 3), dtype=np.uint8)


def shuffle_pairs(gt_imgs, labels, seed=2022):
    """Derangement of the ground truth: same images, wrong pairing.

    Where possible a sample is paired with a DIFFERENT class, so the shuffled
    baseline is a genuine negative control rather than an accidental match.
    """
    rng = np.random.RandomState(seed)
    n = len(gt_imgs)
    idx = np.arange(n)
    for _ in range(100):
        rng.shuffle(idx)
        if all(labels[i] != labels[j] for i, j in zip(range(n), idx)):
            break
    return np.asarray(gt_imgs)[idx], idx
