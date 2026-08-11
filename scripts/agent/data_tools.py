"""ADK 에이전트에 등록할 tool 함수 2종: 데이터 조회, 모델 예측 호출.

ADK는 함수의 타입 힌트와 docstring에서 자동으로 function-calling 스키마를 생성하므로,
각 함수는 명확한 타입과 설명을 갖춰야 한다.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

AGENT_DIR = Path(__file__).resolve().parent
BASE_DIR = AGENT_DIR.parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "annual_revenue_krw", "income_log", "income_per_age", "age", "career_years",
    "biz_operation_years", "biz_location_years", "job_stability_ratio", "is_married", "has_car",
    "revenue_volatility_synth", "debt_to_income_synth",
    "industry_sector_risk_te", "biz_city_risk_te", "biz_region_risk_te",
    "biz_premise_ownership_owned", "biz_premise_ownership_rented", "biz_premise_ownership_norent_noown",
]

_model = None
_thresholds = None
_all_rows = None
_explainer = None


def _load_state():
    global _model, _thresholds, _all_rows
    if _model is None:
        _model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
        _thresholds = json.loads((MODELS_DIR / "threshold_config.json").read_text(encoding="utf-8"))
        _all_rows = pd.concat(
            [pd.read_csv(PROCESSED_DIR / f"{name}.csv") for name in ("train", "val", "test")],
            ignore_index=True,
        )
    return _model, _thresholds, _all_rows


def _get_explainer(model) -> shap.TreeExplainer:
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(model)
    return _explainer


def _to_native(value):
    """numpy 스칼라(bool_/integer/floating)를 JSON 직렬화 가능한 파이썬 기본 타입으로 변환."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def get_applicant_data(applicant_id: int) -> dict:
    """신청자 ID로 정형 데이터(매출, 나이, 업력, 지역/업종 위험도 등)를 조회한다.

    Args:
        applicant_id: 신청자 고유 ID.

    Returns:
        신청자의 피처 값과 실제 라벨(있는 경우)을 담은 dict. 존재하지 않으면 error 키를 담아 반환.
    """
    _, _, all_rows = _load_state()
    row = all_rows[all_rows["id"] == applicant_id]
    if row.empty:
        return {"error": f"applicant_id {applicant_id} 를 찾을 수 없습니다."}
    return row.iloc[0].to_dict()


def predict_risk(applicant_id: int) -> dict:
    """학습된 XGBoost 모델로 신청자의 부도확률을 예측하고 정량 등급을 판정한다.

    Args:
        applicant_id: 신청자 고유 ID.

    Returns:
        default_probability(부도확률), quant_tier(정량 등급: approve/conditional/reject),
        threshold 정보를 담은 dict.
    """
    model, thresholds, all_rows = _load_state()
    row = all_rows[all_rows["id"] == applicant_id]
    if row.empty:
        return {"error": f"applicant_id {applicant_id} 를 찾을 수 없습니다."}

    prob = float(model.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
    if prob < thresholds["t_approve"]:
        tier = "approve"
    elif prob >= thresholds["t_reject"]:
        tier = "reject"
    else:
        tier = "conditional"

    return {
        "applicant_id": applicant_id,
        "default_probability": round(prob, 4),
        "quant_tier": tier,
        "t_approve": thresholds["t_approve"],
        "t_reject": thresholds["t_reject"],
    }


def explain_prediction(applicant_id: int, top_k: int = 5) -> dict:
    """SHAP으로 이 신청자의 부도확률 예측에 각 피처가 얼마나/어느 방향으로 기여했는지 설명한다.

    전체 피처 중요도(model_report.md)와 달리, 이 신청자 한 명의 예측치를 개별적으로 분해한
    값이다. contribution이 양수면 그 피처가 부도확률을 높이는 방향으로, 음수면 낮추는 방향으로
    작용했다는 뜻이다 (단위: log-odds, 확률 퍼센트포인트가 아님).

    Args:
        applicant_id: 신청자 고유 ID.
        top_k: 반환할 상위 기여 피처 개수.

    Returns:
        base_value(전체 평균 기준 log-odds), predicted_probability, top_contributors
        (기여도 절대값 기준 상위 top_k개, feature/value/contribution)를 담은 dict.
    """
    model, _, all_rows = _load_state()
    row = all_rows[all_rows["id"] == applicant_id]
    if row.empty:
        return {"error": f"applicant_id {applicant_id} 를 찾을 수 없습니다."}

    X = row[FEATURE_COLUMNS]
    explanation = _get_explainer(model)(X)

    contributors = [
        {
            "feature": feature,
            "value": _to_native(X.iloc[0][feature]),
            "contribution": round(float(shap_value), 4),
        }
        for feature, shap_value in zip(FEATURE_COLUMNS, explanation.values[0])
    ]
    contributors.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    return {
        "applicant_id": applicant_id,
        "base_value": round(float(explanation.base_values[0]), 4),
        "predicted_probability": float(model.predict_proba(X)[:, 1][0]),
        "top_contributors": contributors[:top_k],
    }


if __name__ == "__main__":
    sample_id = 10736
    print(get_applicant_data(sample_id))
    print(predict_risk(sample_id))
    print(explain_prediction(sample_id))
