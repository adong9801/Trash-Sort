"""Test the model that's in the app, picking bins the same way the app does.

    .venv/bin/python evaluate.py

Runs ../public/model/trashsort.onnx on the photos train.py held back for testing, and on TrashNet.
"""

import json

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader

from train import BATCH, EVAL_TF, OUT_LABELS, OUT_MODEL, Photos, collect, report, split


def main():
    labels = json.loads(OUT_LABELS.read_text())
    session = ort.InferenceSession(str(OUT_MODEL))

    def predict_fn(loader):
        probs, truth = [], []
        for x, y in loader:
            for image in x.numpy():
                logits = session.run(None, {"image": image[None]})[0][0]
                exps = np.exp(logits - logits.max())
                probs.append(exps / exps.sum())
            truth += y.tolist()
        return np.array(probs), np.array(truth)

    _, _, test = split(collect(), labels)
    report(predict_fn, DataLoader(Photos(test, EVAL_TF), BATCH, num_workers=8), labels)


if __name__ == "__main__":
    main()
