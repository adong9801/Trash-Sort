"""Throw out mislabeled photos from the web-scraped recycling dataset.

    .venv/bin/python clean.py

Its "soft plastics" folder has fishing lures and soda bottles, and "polystyrene" has foam
insulation and acoustic panels. CLIP, a model that matches images to text, picks the closest
description for each photo, and only photos that match their own folder's descriptions are kept.
Writes data/clean.json, which train.py reads.

(The dataset's glass, aluminium, paper, and hard plastic folders were too messy even after cleaning,
so train.py skips them. The main dataset has plenty of those.)
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DATA = Path(__file__).parent / "data"

# Folder -> descriptions a photo in it should match.
KEEP_IF = {
    "soft plastics": [
        "a plastic shopping bag",
        "a plastic bag",
        "plastic wrap or cling film",
        "a chip bag or snack wrapper",
    ],
    "polystyrene": [
        "a styrofoam cup",
        "a styrofoam takeout container",
        "styrofoam packaging",
    ],
}

# Things these folders are wrongly full of. A photo that matches one of these is dropped.
DISTRACTORS = [
    "a plastic bottle",
    "a plastic food container",
    "a glass bottle",
    "an aluminum drink can",
    "a toy",
    "a fishing lure",
    "a ball",
    "a balloon",
    "a building",
    "a metal beam or pipe",
    "a vase",
    "a lamp",
    "a drinking glass",
    "a decorative pattern or texture",
    "foam insulation",
    "acoustic foam panels",
    "furniture",
    "a person",
    "food",
    "clothing",
    "a plastic folder",
    "a roll of tape or string",
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    prompts = [d for ds in KEEP_IF.values() for d in ds] + DISTRACTORS
    with torch.no_grad():
        text = processor(text=[f"a photo of {p}" for p in prompts], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text).pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    kept = {}
    for folder, targets in KEEP_IF.items():
        paths = sorted((DATA / "viola77data" / folder).glob("*.jpg"))
        kept[folder] = []
        for start in range(0, len(paths), 32):
            batch = paths[start : start + 32]
            images = [Image.open(p).convert("RGB") for p in batch]
            with torch.no_grad():
                inputs = processor(images=images, return_tensors="pt").to(device)
                features = model.get_image_features(**inputs).pooler_output
                features = features / features.norm(dim=-1, keepdim=True)
            best = (features @ text_features.T).argmax(dim=1).cpu().tolist()
            kept[folder] += [p.name for p, b in zip(batch, best) if prompts[b] in targets]
        print(f"{folder:15} kept {len(kept[folder]):3} of {len(paths)}")

    (DATA / "clean.json").write_text(json.dumps(kept, indent=2))


if __name__ == "__main__":
    main()
