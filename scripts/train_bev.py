#현재 시점 BEV만 사용한 CNN. train 내부 주행 구간별 5-fold 평가.
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_dataset
from src.paths import DATASET_DIR, RESULTS_DIR

EPOCHS = 3
BATCH_SIZE = 128
SEED = 1729


def make_model():
    # 위치 관계를 보존하도록 마지막 특징 맵을 3×6 격자로 요약한다.
    layers = []
    for channels_in, channels_out in [(5, 12), (12, 24), (24, 32)]:
        layers.extend([nn.Conv2d(channels_in, channels_out, 3, stride=2, padding=1),
                       nn.BatchNorm2d(channels_out), nn.ReLU()])
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 6)), nn.Flatten(),
                         nn.Linear(32 * 3 * 6, 32), nn.ReLU(), nn.Linear(32, 1))


def main():
    data = load_dataset(DATASET_DIR)
    indices = np.flatnonzero(data.mask("train"))
    labels = data.y[indices]
    groups = [data.rows[i]["segment_id"] for i in indices]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    scores = np.empty(len(labels))
    fold_ap = []

    def images(rows):
        return torch.from_numpy(data.bev(indices[rows])).to(device, dtype=torch.float32)

    for fold, (fit, held) in enumerate(GroupKFold(5).split(indices, labels, groups)):
        torch.manual_seed(SEED + fold)
        rng = np.random.default_rng(SEED + fold)
        model = make_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        positives = fit[labels[fit] == 1]
        negatives = fit[labels[fit] == 0]
        for _ in range(EPOCHS):
            # 학습에서만 음성을 다시 추출한다. 평가에서는 전체 표본을 사용한다.
            sample = rng.choice(negatives, min(4 * len(positives), len(negatives)), replace=False)
            rows = rng.permutation(np.concatenate([positives, sample]))
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start:start + BATCH_SIZE]
                logits = model(images(batch)).flatten()
                y = torch.tensor(labels[batch], device=device, dtype=torch.float32)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(held), BATCH_SIZE):
                batch = held[start:start + BATCH_SIZE]
                scores[batch] = model(images(batch)).flatten().sigmoid().cpu().numpy()
        fold_ap.append(float(average_precision_score(labels[held], scores[held])))
        print(f"Fold {fold + 1}: AP {fold_ap[-1]:.4f}", flush=True)

    result = dict(train_samples=len(labels), positives=int(labels.sum()), epochs=EPOCHS,
                  train_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                  fold_ap=fold_ap, pooled_ap=float(average_precision_score(labels, scores)))
    folder = RESULTS_DIR / "bev"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "train_cv.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
