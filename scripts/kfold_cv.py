"""K-fold 교차검증 — 보고된 test AUC가 특정 분할에 우연히 좌우된 값은 아닌지 분산을 확인한다.

train_model.py는 train/val/test 한 번의 분할만 사용해서, 보고된 test AUC(models/metrics_report.json의
test_auc)가 그 특정 분할에서 우연히 잘/못 나온 건 아닌지 확인할 방법이 없었다. 이 스크립트는 동일한
FEATURE_COLUMNS/하이퍼파라미터로 5-fold stratified CV를 돌려 fold별 AUC의 평균과 표준편차를 산출한다.

주의(참고용 단순화): 재사용하는 processed/*.csv의 *_risk_te 피처는 원래 train split에서만 fit된
target encoding 값이 이미 고정되어 있다(preprocess.py 참고, val/test로의 누수는 없음). 이 스크립트는
그 값을 그대로 재사용해 CV를 도는 것이라, target encoding까지 fold마다 다시 fit하는 완전한 nested CV는
아니다 — "이 피처셋으로 분할을 바꿔가며 재현되는 성능인가"를 보는 용도로는 충분하다.

실행:
    python kfold_cv.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_model import FEATURE_COLUMNS, LABEL_COLUMN, MODELS_DIR, PROCESSED_DIR, SEED, train_xgboost

N_SPLITS = 5


def load_all_data():
    df = pd.concat(
        [pd.read_csv(PROCESSED_DIR / f"{name}.csv") for name in ("train", "val", "test")],
        ignore_index=True,
    )
    return df[FEATURE_COLUMNS], df[LABEL_COLUMN]


def main():
    X, y = load_all_data()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_aucs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = train_xgboost(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        fold_aucs.append(auc)
        print(f"Fold {fold_idx}: AUC = {auc:.4f}")

    fold_aucs = np.array(fold_aucs)
    mean_auc = float(fold_aucs.mean())
    std_auc = float(fold_aucs.std(ddof=1))

    result = {
        "n_splits": N_SPLITS,
        "fold_aucs": [float(a) for a in fold_aucs],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "min_auc": float(fold_aucs.min()),
        "max_auc": float(fold_aucs.max()),
        "reported_test_auc_single_split": None,
    }

    metrics_path = MODELS_DIR / "metrics_report.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
        result["reported_test_auc_single_split"] = existing.get("test_auc")

    with open(MODELS_DIR / "kfold_cv_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== K-fold CV 결과 ===")
    print(f"Fold별 AUC: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"평균: {mean_auc:.4f} (표준편차 {std_auc:.4f})")
    print(f"범위: {fold_aucs.min():.4f} ~ {fold_aucs.max():.4f}")
    if result["reported_test_auc_single_split"] is not None:
        print(f"기존 단일 test-split AUC: {result['reported_test_auc_single_split']:.4f}")
    print(f"결과 저장: {MODELS_DIR / 'kfold_cv_report.json'}")


if __name__ == "__main__":
    main()
