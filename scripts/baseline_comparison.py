"""단순 베이스라인(로지스틱 회귀)과 XGBoost/LightGBM 비교.

기존 train_model.py는 XGBoost vs LightGBM만 val AUC로 비교했다. "트리 부스팅이 정말 필요한가"를
보려면 단순한 선형 모델과의 비교가 필요하다. 이 스크립트는 같은 split/피처/시드로
로지스틱 회귀를 학습해 세 모델을 같은 기준으로 비교한다.

- 학습/선택/평가 분리는 train_model.py와 동일하다: val AUC로 모델·하이퍼파라미터를 고르고,
  test(held-out)는 최종 평가에만 쓴다. 로지스틱 회귀의 규제 강도 C만 val AUC로 고른다
  (XGBoost/LightGBM은 train_model.py의 고정 하이퍼파라미터를 그대로 쓴다 — 튜닝하지 않았다).
- 같은 임계값 탐색(find_thresholds)을 적용해 승인/조건부승인/거절 티어 성능도 비교한다.
- 5-fold CV(kfold_cv.py와 같은 폴드)로 분할 의존성도 확인한다.
- models/ 의 서빙 모델, 임계값, metrics_report.json은 건드리지 않는다.
  결과는 models/baseline_comparison_report.json 에만 쓴다.

실행:
    python baseline_comparison.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_model import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    MODELS_DIR,
    PROCESSED_DIR,
    SEED,
    apply_tiers,
    find_thresholds,
    load_split,
    tier_summary,
    train_lightgbm,
    train_xgboost,
)

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
N_SPLITS = 5
N_BOOTSTRAP = 500
REPORT_PATH = MODELS_DIR / "baseline_comparison_report.json"


def make_logreg(C: float):
    """스케일링 + 클래스 불균형 보정(class_weight=balanced; XGBoost/LightGBM과 같은 취지)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def train_logreg(X_train, y_train, C: float):
    model = make_logreg(C)
    model.fit(X_train, y_train)
    return model


def reject_tier_metrics(probs_val, y_val, probs_test, y_test) -> dict:
    """train_model.py와 같은 방식으로 val에서 임계값을 찾고, test의 티어별 부도율과 거절 기준 P/R을 낸다."""
    thr = find_thresholds(probs_val, y_val.to_numpy())
    tiers = apply_tiers(probs_test, thr["t_approve"], thr["t_reject"])
    summary = tier_summary(tiers, y_test.to_numpy())
    pred_reject = tiers == "reject"
    y = y_test.to_numpy()
    tp = int((pred_reject & (y == 1)).sum())
    fp = int((pred_reject & (y == 0)).sum())
    fn = int((~pred_reject & (y == 1)).sum())
    return {
        "thresholds": {k: thr[k] for k in ("t_approve", "t_reject", "relaxation_step")},
        "test_tier_bad_rate": {t: float(summary.loc[t, "bad_rate"]) for t in summary.index},
        "test_tier_share": {t: float(summary.loc[t, "share"]) for t in summary.index},
        "reject_precision": tp / (tp + fp) if tp + fp else None,
        "reject_recall": tp / (tp + fn) if tp + fn else None,
    }


def paired_bootstrap_auc_diff(y, p_a, p_b, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """같은 test 샘플을 재추출해 AUC(a) - AUC(b)의 95% 구간을 구한다 (a=XGBoost, b=비교 대상)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "n_boot": int(len(diffs)),
    }


def kfold_aucs(X, y, logreg_C: float) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)  # kfold_cv.py와 같은 폴드
    out = {"xgboost": [], "lightgbm": [], "logistic_regression": []}
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        out["xgboost"].append(roc_auc_score(y_te, train_xgboost(X_tr, y_tr).predict_proba(X_te)[:, 1]))
        out["lightgbm"].append(roc_auc_score(y_te, train_lightgbm(X_tr, y_tr).predict_proba(X_te)[:, 1]))
        out["logistic_regression"].append(
            roc_auc_score(y_te, train_logreg(X_tr, y_tr, logreg_C).predict_proba(X_te)[:, 1])
        )
    return {
        k: {
            "fold_aucs": [float(a) for a in v],
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)),
        }
        for k, v in out.items()
    }


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, _ = load_split("test")

    # --- 1. 로지스틱 회귀: C를 val AUC로 선택 ---
    c_val_aucs = {}
    for C in C_GRID:
        m = train_logreg(X_train, y_train, C)
        c_val_aucs[C] = float(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
    best_C = max(c_val_aucs, key=c_val_aucs.get)
    print("로지스틱 회귀 C별 val AUC:", {str(k): round(v, 4) for k, v in c_val_aucs.items()}, "→ 선택 C =", best_C)

    models = {
        "xgboost": train_xgboost(X_train, y_train),
        "lightgbm": train_lightgbm(X_train, y_train),
        "logistic_regression": train_logreg(X_train, y_train, best_C),
    }
    val_probs = {k: m.predict_proba(X_val)[:, 1] for k, m in models.items()}
    test_probs = {k: m.predict_proba(X_test)[:, 1] for k, m in models.items()}

    results = {}
    for name in models:
        results[name] = {
            "val_auc": float(roc_auc_score(y_val, val_probs[name])),
            "test_auc": float(roc_auc_score(y_test, test_probs[name])),
            "tiers": reject_tier_metrics(val_probs[name], y_val, test_probs[name], y_test),
        }

    # --- 2. XGBoost 대비 test AUC 차이 (같은 test 샘플 재추출) ---
    diffs = {
        f"xgboost_minus_{k}": paired_bootstrap_auc_diff(y_test, test_probs["xgboost"], test_probs[k])
        for k in ("lightgbm", "logistic_regression")
    }

    # --- 3. 5-fold CV ---
    all_X = pd.concat([pd.read_csv(PROCESSED_DIR / f"{n}.csv") for n in ("train", "val", "test")], ignore_index=True)
    cv = kfold_aucs(all_X[FEATURE_COLUMNS], all_X[LABEL_COLUMN], best_C)

    # --- 4. 로지스틱 회귀 계수 (표준화 기준, 해석용) ---
    coefs = models["logistic_regression"].named_steps["logisticregression"].coef_[0]
    coef_series = pd.Series(coefs, index=FEATURE_COLUMNS).sort_values(key=np.abs, ascending=False)

    report = {
        "selection_rule": "val AUC로 LR의 C만 선택. XGBoost/LightGBM은 train_model.py 고정 하이퍼파라미터(튜닝 안 함). test는 최종 평가 전용.",
        "logistic_regression_config": {
            "preprocessing": "StandardScaler",
            "class_weight": "balanced",
            "penalty": "l2",
            "C_grid_val_auc": {str(k): v for k, v in c_val_aucs.items()},
            "selected_C": best_C,
        },
        "results": results,
        "test_auc_difference_bootstrap": diffs,
        "kfold_cv": {"n_splits": N_SPLITS, **cv},
        "logistic_regression_standardized_coefficients": {k: float(v) for k, v in coef_series.items()},
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== 모델 비교 (같은 split/피처/시드) ===")
    print(f"{'모델':<22}{'Val AUC':>9}{'Test AUC':>10}{'5-fold CV(평균±표준편차)':>26}{'거절 Precision':>16}{'거절 Recall':>13}")
    for name in models:
        r, c = results[name], cv[name]
        t = r["tiers"]
        print(f"{name:<22}{r['val_auc']:>9.4f}{r['test_auc']:>10.4f}"
              f"{c['mean']:>16.4f} ± {c['std']:.4f}{t['reject_precision']:>16.3f}{t['reject_recall']:>13.3f}")
    for k, d in diffs.items():
        print(f"{k}: test AUC 차이 {d['mean_diff']:+.4f} (95% 부트스트랩 구간 {d['ci95'][0]:+.4f} ~ {d['ci95'][1]:+.4f})")
    print(f"\n저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
