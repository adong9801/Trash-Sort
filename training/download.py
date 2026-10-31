"""Download the training photos into data/ (about 24,000 small images, takes ~20 minutes).

    .venv/bin/python download.py
"""

import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

DATA = Path(__file__).parent / "data"

# MIT and Apache-2.0 licensed.
for repo in ["omasteam/waste-garbage-management-dataset", "viola77data/recycling-dataset"]:
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=DATA / repo.split("/")[0], max_workers=16)

# ImageNet's "plastic bag" photos. ImageNet allows non-commercial research and educational use.
bags = DATA / "imagenet_plastic_bag"
bags.mkdir(parents=True, exist_ok=True)
for split in ["train", "validation"]:
    path = hf_hub_download(
        "mlnomad/imnet1k_plastic_bag", f"data/{split}-00000-of-00001.parquet", repo_type="dataset", local_dir=DATA / "hf_cache"
    )
    for i, row in enumerate(pq.read_table(path).to_pylist()):
        (bags / f"{split}_{i:05}.jpg").write_bytes(row["image"]["bytes"])

# Only used for testing, never for training.
zip_path = hf_hub_download("garythung/trashnet", "dataset-resized.zip", repo_type="dataset", local_dir=DATA / "trashnet")
zipfile.ZipFile(zip_path).extractall(DATA / "trashnet")
