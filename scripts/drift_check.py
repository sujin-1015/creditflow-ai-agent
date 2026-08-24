"""예측 확률 분포 드리프트 체크 (PSI, Population Stability Index).

이 프로젝트는 정적 Kaggle 데이터셋 기반 PoC라 "시간에 따라 유입되는 실제 신규 데이터"가
없다. 그래서 "지난달 대비 이번달 신청자 분포가 얼마나 달라졌는지"를 측정하는 전형적인
운영 드리프트 체크는 이 프로젝트에 그대로 적용할 수 없다.

대신 실질적으로 의미 있는 질문으로 바꿨다: **재학습된 후보 모델의 예측 확률 분포가 현재
배포된(프로덕션) 모델과 같은 test set에서 얼마나 달라졌는가?** AUC가 비슷하더라도 확률
분포 자체가 크게 밀리면, `threshold_config.json`에 고정된 t_approve/t_reject(특정 확률
값)가 더 이상 원래 의도한 승인군 부도율 8% 이하 / 거절군 20% 이상 제약을 만족하지 못할 수
있다 — model_registry.py의 AUC 게이트가 놓칠 수 있는 문제를 이 체크가 보완한다.

PSI 해석 (업계 통용 기준):
    PSI < 0.1  : 유의미한 변화 없음
    0.1 <= PSI < 0.25 : 중간 수준 변화, 모니터링 권장
    PSI >= 0.25 : 큰 변화, threshold 재산정 등 조사 필요

실행:
    python drift_check.py                 # 현재 프로덕션 모델 vs 후보(models/_candidate) 비교
    python drift_check.py --self-check     # 현재 프로덕션 모델을 자기 자신과 비교 (PSI=0 sanity check)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "processed"
DRIFT_HISTORY_PATH = MODELS_DIR / "drift_history.json"

FEATURE_COLUMNS = [
    "annual_revenue_krw", "income_log", "income_per_age", "age", "career_years",
    "biz_operation_years", "biz_location_years", "job_stability_ratio", "is_married", "has_car",
    "revenue_volatility_synth", "debt_to_income_synth",
    "industry_sector_risk_te", "biz_city_risk_te", "biz_region_risk_te",
    "biz_premise_ownership_owned", "biz_premise_ownership_rented", "biz_premise_ownership_norent_noown",
]


def compute_psi(baseline_probs: np.ndarray, candidate_probs: np.ndarray, n_bins: int = 10) -> dict:
    """baseline(기준) 대비 candidate(비교 대상) 확률 분포의 PSI를 계산한다.
    구간은 baseline 분포의 분위수로 나눈다 — 두 분포가 완전히 같으면 각 구간에 정확히
    1/n_bins씩 들어가므로 PSI=0이 나온다."""
    edges = np.unique(np.percentile(baseline_probs, np.linspace(0, 100, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf  # 범위 밖 값도 양 끝 구간에 포함

    base_counts, _ = np.histogram(baseline_probs, bins=edges)
    cand_counts, _ = np.histogram(candidate_probs, bins=edges)

    base_pct = base_counts / base_counts.sum()
    cand_pct = cand_counts / cand_counts.sum()

    # 0으로 나누기/log(0) 방지용 최소값
    eps = 1e-4
    base_pct = np.clip(base_pct, eps, None)
    cand_pct = np.clip(cand_pct, eps, None)

    psi_per_bin = (cand_pct - base_pct) * np.log(cand_pct / base_pct)
    psi = float(psi_per_bin.sum())

    if psi < 0.1:
        severity = "no_significant_change"
    elif psi < 0.25:
        severity = "moderate_change_monitor"
    else:
        severity = "significant_change_investigate"

    return {
        "psi": round(psi, 4),
        "severity": severity,
        "n_bins": len(edges) - 1,
        "baseline_n": len(baseline_probs),
        "candidate_n": len(candidate_probs),
    }


def _predict_test_probs(model_path: Path) -> np.ndarray:
    model = joblib.load(model_path)
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]


def _append_history(entry: dict) -> None:
    history = json.loads(DRIFT_HISTORY_PATH.read_text(encoding="utf-8")) if DRIFT_HISTORY_PATH.exists() else {"checks": []}
    history["checks"].append(entry)
    DRIFT_HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", help="프로덕션 모델을 자기 자신과 비교 (PSI=0 sanity check)")
    args = parser.parse_args()

    baseline_path = MODELS_DIR / "xgboost_model.joblib"
    if not baseline_path.exists():
        print("프로덕션 모델(models/xgboost_model.joblib)이 없습니다.")
        return 1

    if args.self_check:
        candidate_path = baseline_path
        label = "self-check"
    else:
        candidate_path = MODELS_DIR / "_candidate" / "xgboost_model.joblib"
        if not candidate_path.exists():
            print("비교할 후보 모델(models/_candidate/xgboost_model.joblib)이 없습니다 — "
                  "train_model.py를 먼저 실행하거나 --self-check로 sanity check만 해보세요.")
            return 1
        label = "candidate_vs_production"

    baseline_probs = _predict_test_probs(baseline_path)
    candidate_probs = _predict_test_probs(candidate_path)
    result = compute_psi(baseline_probs, candidate_probs)
    result["check_type"] = label

    print(f"PSI: {result['psi']} ({result['severity']})")
    if result["severity"] != "no_significant_change":
        print("  [경고] 예측 확률 분포가 유의미하게 달라졌습니다 — threshold_config.json 재산정을 검토하세요.")

    from datetime import datetime, timezone
    _append_history({"timestamp": datetime.now(timezone.utc).isoformat(), **result})
    print(f"기록 저장: {DRIFT_HISTORY_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
