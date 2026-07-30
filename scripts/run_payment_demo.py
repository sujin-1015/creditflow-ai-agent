"""
모델 판정 -> (모의) 온체인 집행까지 이어지는 end-to-end 데모.

test.csv에서 표본을 뽑아 XGBoost 확률 -> 3단계 판정 -> disburse_loan() 순서로 실행하고
결과를 콘솔에 출력 + onchain/payments_log.json 에 누적 기록한다.

실행:
    python run_payment_demo.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd

import payment_mock
from payment_mock import disburse_loan

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "annual_revenue_krw", "income_log", "income_per_age", "age", "career_years",
    "biz_operation_years", "biz_location_years", "job_stability_ratio", "is_married", "has_car",
    "revenue_volatility_synth", "debt_to_income_synth",
    "industry_sector_risk_te", "biz_city_risk_te", "biz_region_risk_te",
    "biz_premise_ownership_owned", "biz_premise_ownership_rented", "biz_premise_ownership_norent_noown",
]

LOAN_RATE = 0.05       # 연매출의 5%를 소액대출 한도로 산정 (PoC 단순 규칙)
LOAN_CAP_KRW = 5_000_000


def decide_tier(prob: float, t_approve: float, t_reject: float) -> str:
    if prob < t_approve:
        return "approve"
    if prob >= t_reject:
        return "reject"
    return "conditional"


def make_rationale(row: pd.Series, prob: float, tier: str) -> str:
    tier_kr = {"approve": "승인", "conditional": "조건부승인", "reject": "거절"}[tier]
    return (
        f"예측 부도확률 {prob:.1%} (업종 위험도 {row['industry_sector_risk_te']:.1%}, "
        f"지역 위험도 {row['biz_city_risk_te']:.1%}, 업력 {row['biz_operation_years']}년, "
        f"연매출 {row['annual_revenue_krw']:,.0f}) 기준 {tier_kr} 판정."
    )


def main():
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    with open(MODELS_DIR / "threshold_config.json", encoding="utf-8") as f:
        thresholds = json.load(f)

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    sample = test_df.sample(n=10, random_state=7).reset_index(drop=True)

    probs = model.predict_proba(sample[FEATURE_COLUMNS])[:, 1]

    print(f"{'ID':>8} {'실제라벨':>8} {'예측확률':>8} {'판정':>10} {'집행상태':>10} {'tx_signature'}")
    for i, row in sample.iterrows():
        prob = probs[i]
        tier = decide_tier(prob, thresholds["t_approve"], thresholds["t_reject"])
        rationale = make_rationale(row, prob, tier)
        loan_amount = min(int(row["annual_revenue_krw"] * LOAN_RATE), LOAN_CAP_KRW)

        result = disburse_loan(
            applicant_id=int(row["id"]),
            decision=tier,
            requested_loan_krw=loan_amount,
            rationale=rationale,
        )

        sig_display = result.tx_signature[:20] + "..." if result.tx_signature else "-"
        tier_kr = {"approve": "승인", "conditional": "조건부", "reject": "거절"}[tier]
        print(f"{result.applicant_id:>8} {int(row['default_risk_label']):>8} {prob:>7.1%} "
              f"{tier_kr:>10} {result.status:>10} {sig_display}")

    print(f"\n로그 저장 위치: {BASE_DIR / 'onchain' / 'payments_log.json'}")
    if payment_mock.MOCK_MODE:
        print("(MOCK_MODE=True — 실제 네트워크 호출 없음, 로컬 시뮬레이션)")
    else:
        print("(MOCK_MODE=False — 승인 건은 공개 Solana devnet에 실제 트랜잭션 실행)")


if __name__ == "__main__":
    main()
