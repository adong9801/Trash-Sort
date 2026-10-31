"""Train the TrashSort image model and export it for the app.

    .venv/bin/python train.py

Reads photos from data/ (see download.py and clean.py), fine-tunes MobileNetV3 on them, prints
test results, and writes ../public/model/trashsort.onnx plus the label list the app uses.
Takes about 15 minutes on an M-series Mac.
"""

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import certifi
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

# Python from python.org on macOS can't verify HTTPS downloads (of the pretrained weights) without this.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT_MODEL = ROOT.parent / "public" / "model" / "trashsort.onnx"
OUT_LABELS = ROOT.parent / "src" / "lib" / "model-labels.json"

# Folder in each source dataset -> the item type the model learns.
SOURCES = {
    "omasteam": {
        "battery": "battery",
        "biological": "food",
        "cardboard": "cardboard",
        "clothes": "clothes",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "shoes": "shoes",
        "trash": "trash",
    },
    # Web-scraped and messy: only the folders that hold what they say, and for the two listed in
    # data/clean.json, only the photos clean.py kept.
    "viola77data": {
        "batteries": "battery",
        "cardboard": "cardboard",
        "disposable plates": "disposable_plate",
        "paper towel": "paper_towel",
        "polystyrene": "styrofoam",
        "soft plastics": "plastic_film",
        "takeaway cups": "takeaway_cup",
    },
    "imagenet_plastic_bag": {".": "plastic_film"},
}

# TrashNet is held out entirely: different photographers and backgrounds, so it shows
# how well the model copes with photos unlike the ones it trained on.
TRASHNET_BINS = {
    "cardboard": "recycling",
    "glass": "recycling",
    "metal": "recycling",
    "paper": "recycling",
    "plastic": "recycling",
    "trash": "trash",
}

SIZE = 224
SEED = 0
EPOCHS = 12
BATCH = 64
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Must match the bin each item maps to in src/lib/items.ts.
ITEM_BINS = {
    "battery": "special",
    "cardboard": "recycling",
    "clothes": "special",
    "disposable_plate": "trash",
    "food": "compost",
    "glass": "recycling",
    "metal": "recycling",
    "paper": "recycling",
    "paper_towel": "compost",
    "plastic": "recycling",
    "plastic_film": "trash",
    "shoes": "special",
    "styrofoam": "trash",
    "takeaway_cup": "trash",
    "trash": "trash",
}

# Your own photos: data/school/<item type>/, e.g. data/school/plastic_film/bag1.jpg.
# Real photos from your school teach the model the most.
SOURCES["school"] = {item: item for item in ITEM_BINS}
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}


class SquareCrop:
    """Center-crop to a square, the same way the app crops photos before resizing."""

    def __call__(self, img):
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        return img.crop((left, top, left + side, top + side))


EVAL_TF = T.Compose([SquareCrop(), T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
# Heavy variation, so the model learns what the item looks like rather than the photo's background
# or lighting (many internet photos of batteries, for example, are product shots on white).
TRAIN_TF = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.3, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(0.4, 0.4, 0.4, 0.05),
    T.RandomGrayscale(0.1),
    T.RandomApply([T.GaussianBlur(5)], p=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25),
])


class Photos(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def looks_like(path):
    """A fingerprint of what a photo looks like, the same even after resizing or recompressing."""
    pixels = np.asarray(Image.open(path).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return hashlib.md5(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes()).hexdigest()


def trashnet_photos():
    return [
        (p, folder)
        for folder in TRASHNET_BINS
        for p in sorted((DATA / "trashnet" / "dataset-resized" / folder).glob("*.jpg"))
    ]


def collect():
    """All (path, item) pairs, without repeats of the same photo, so none can be in both the
    training and test photos. The main dataset includes TrashNet's photos, so those are left out
    too, to keep TrashNet a fair test."""
    cleaned = json.loads((DATA / "clean.json").read_text())
    seen = {looks_like(p) for p, _ in trashnet_photos()}
    items = []
    for source, folders in SOURCES.items():
        for folder, item in folders.items():
            photos = (p for p in (DATA / source / folder).glob("*") if p.suffix.lower() in PHOTO_TYPES)
            for path in sorted(photos):
                if source == "viola77data" and folder in cleaned and path.name not in cleaned[folder]:
                    continue
                fingerprint = looks_like(path)
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    items.append((path, item))
    return items


def split(items, labels):
    by_item = defaultdict(list)
    for path, item in items:
        by_item[item].append((path, labels.index(item)))
    rng = random.Random(SEED)
    train, val, test = [], [], []
    for group in by_item.values():
        rng.shuffle(group)
        n_test = n_val = max(1, len(group) // 10)
        test += group[:n_test]
        val += group[n_test : n_test + n_val]
        train += group[n_test + n_val :]
    return train, val, test


def predict(model, loader, device):
    """Each photo's probability for every item type, plus the right answers."""
    model.eval()
    probs, truth = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(model(x.to(device)).softmax(1).cpu().numpy())
            truth += y.tolist()
    return np.concatenate(probs), np.array(truth)


def app_bin(probs, labels):
    """The bin the app shows: it adds up the item types in each bin and picks the biggest total
    (see src/lib/classifier.web.ts)."""
    bins = sorted(set(ITEM_BINS.values()))
    totals = [sum(p for p, label in zip(probs, labels) if ITEM_BINS[label] == b) for b in bins]
    return bins[int(np.argmax(totals))]


def main():
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    items = collect()
    labels = sorted({item for _, item in items})
    assert set(labels) == set(ITEM_BINS), "every item needs a bin"
    train, val, test = split(items, labels)
    counts = Counter(labels[y] for _, y in train)
    print(f"{len(items)} photos, {len(labels)} item types: train {len(train)}, val {len(val)}, test {len(test)}")
    for label in labels:
        print(f"  {label:17} {counts[label]}")

    # Rare item types (a few dozen photos) would otherwise be drowned out by common ones.
    # Capped, so the rarest ones don't turn into the model's default guess.
    weights = torch.tensor([min(3.0, (len(train) / (len(labels) * counts[l])) ** 0.5) for l in labels])
    loaders = {
        "train": DataLoader(Photos(train, TRAIN_TF), BATCH, shuffle=True, num_workers=8, persistent_workers=True),
        "val": DataLoader(Photos(val, EVAL_TF), BATCH, num_workers=8, persistent_workers=True),
        "test": DataLoader(Photos(test, EVAL_TF), BATCH, num_workers=8),
    }

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(labels))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    # The pretrained part learns slowly, so it keeps what it knows about everyday objects in
    # general; the new last layer learns fast.
    lrs = [2e-4, 1e-3]
    optimizer = torch.optim.AdamW(
        [{"params": model.features.parameters()}, {"params": model.classifier.parameters()}], weight_decay=0.05
    )
    steps = EPOCHS * len(loaders["train"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lrs, total_steps=steps, pct_start=0.1)

    best_acc, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        probs, truth = predict(model, loaders["val"], device)
        acc = (probs.argmax(1) == truth).mean()
        print(f"epoch {epoch + 1:2}: loss {total / len(loaders['train']):.3f}, val accuracy {acc:.1%}", flush=True)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    report(lambda loader: predict(model, loader, device), loaders["test"], labels)
    export(model.cpu(), labels)


def report(predict_fn, test_loader, labels):
    """predict_fn(loader) returns (probabilities, right answers), like predict()."""
    probs, truth = predict_fn(test_loader)
    preds = probs.argmax(1)
    print(f"\nTest set ({len(truth)} photos it never trained on)")
    print(f"  item type correct: {(preds == truth).mean():.1%}")
    print(f"  bin correct:       {np.mean([app_bin(p, labels) == ITEM_BINS[labels[t]] for p, t in zip(probs, truth)]):.1%}")
    for i, label in enumerate(labels):
        mask = truth == i
        wrong = Counter(labels[p] for p in preds[mask] if p != i).most_common(2)
        print(f"  {label:17} {(preds[mask] == i).mean():6.1%} of {mask.sum():4}   most often confused with {wrong}")

    trashnet = trashnet_photos()
    loader = DataLoader(Photos([(p, 0) for p, _ in trashnet], EVAL_TF), BATCH, num_workers=8)
    probs, _ = predict_fn(loader)
    right = defaultdict(list)
    for (_, folder), p in zip(trashnet, probs):
        right[folder].append(app_bin(p, labels) == TRASHNET_BINS[folder])
    print(f"\nTrashNet ({len(trashnet)} photos from a different dataset)")
    print(f"  bin correct: {np.mean([r for rs in right.values() for r in rs]):.1%}")
    for folder, rs in right.items():
        print(f"  {folder:10} {np.mean(rs):6.1%} -> {TRASHNET_BINS[folder]}")


def export(model, labels):
    model.eval()
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model, (dummy,), OUT_MODEL, input_names=["image"], output_names=["logits"],
        opset_version=18, dynamo=True, external_data=False,
    )
    OUT_LABELS.write_text(json.dumps(labels, indent=2) + "\n")

    import onnxruntime as ort

    session = ort.InferenceSession(str(OUT_MODEL))
    with torch.no_grad():
        expected = model(dummy).numpy()
    got = session.run(None, {"image": dummy.numpy()})[0]
    print(f"\nExported {OUT_MODEL} ({OUT_MODEL.stat().st_size / 1e6:.1f} MB), max difference vs PyTorch {np.abs(got - expected).max():.2e}")


if __name__ == "__main__":
    main()
