"""트리 부스팅 설정 선택과 주행 구간 단위 bootstrap 신뢰구간."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

# (학습률, 잎 수, 잎당 최소 표본, 반복 수) 후보. validation AP가 가장 높은 설정을 쓴다.
CANDIDATES = ((0.07, 15, 30, 200), (0.05, 7, 30, 300), (0.05, 15, 20, 400),
              (0.03, 31, 30, 400), (0.07, 15, 60, 200), (0.10, 7, 20, 150))


def boosting(lr, leaves, min_leaf, iters, class_weight=None):
    return HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
                                          max_iter=iters, l2_regularization=1.0, class_weight=class_weight,
                                          early_stopping=False, random_state=1729)


def fit_select(features, y, train, val, class_weight=None):
    # 학습은 train만, 설정과 분류 기준은 validation만 사용한다.
    best = None
    for config in CANDIDATES:
        model = boosting(*config, class_weight=class_weight).fit(features[train], y[train])
        scores = model.predict_proba(features[val])[:, 1]
        ap = float(average_precision_score(y[val], scores))
        if best is None or ap > best[0]:
            best = (ap, config, model, scores)
    return best[1:]


def bootstrap_f1(y, scores, threshold, group_ids, repeats=1000):
    # 주행 구간(segment_id) 단위로 다시 뽑아 F1이 표본에 따라 얼마나 흔들리는지 본다.
    _, group = np.unique(group_ids, return_inverse=True)
    rng = np.random.default_rng(1729)
    pred, values = scores >= threshold, []
    for _ in range(repeats):
        w = np.bincount(rng.integers(group.max() + 1, size=group.max() + 1), minlength=group.max() + 1)[group]
        tp, fp, fn = (w * (pred & (y == 1))).sum(), (w * (pred & (y == 0))).sum(), (w * (~pred & (y == 1))).sum()
        if tp + fp + fn:
            values.append(2 * tp / (2 * tp + fp + fn))
    return np.percentile(values, [2.5, 97.5]).tolist()
