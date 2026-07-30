"""최종 통합 데모 — Gemini 에이전트 판정 -> (승인 시) 실제 devnet 자동 집행까지.

흐름 (신청자 1명당):
    1. decision.make_final_decision(applicant_id)
       - predict_risk(정량) + 사업자 설명 요약(정성) + RAG(정책) -> Gemini function calling 최종 판정
    2. report.build_rationale_report(...) -> 근거 리포트 저장 (onchain/rationale_reports/)
    3. payment_mock.disburse_loan(...) -> 승인이면 실제 Solana devnet 트랜잭션 집행,
       조건부/거절이면 근거만 기록 (onchain/payments_log.json)

무료 API 할당량 보호를 위해 사업자 설명 텍스트가 준비된 4명(승인/조건부->승인 상향/거절 케이스 포함)만 실행한다.
"""

import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))  # payment_mock, devnet_transfer 임포트용

from decision import make_final_decision  # noqa: E402
from report import build_rationale_report  # noqa: E402
from business_text import SAMPLE_BUSINESS_DESCRIPTIONS  # noqa: E402
import payment_mock  # noqa: E402

REPORT_DIR = SCRIPTS_DIR.parent / "onchain" / "rationale_reports"
DECISION_KR = {"approve": "승인", "conditional": "조건부승인", "reject": "거절"}


def run_one(applicant_id: int) -> None:
    print(f"\n{'=' * 60}\n신청자 {applicant_id} 심사 시작\n{'=' * 60}")

    decision_result = make_final_decision(applicant_id)
    print(f"정량 등급: {decision_result.quant_tier} (부도확률 {decision_result.default_probability:.1%})")
    print(
        f"최종 판정: {decision_result.final_decision}"
        f" (조정: {decision_result.adjustment_applied}/{decision_result.adjustment_direction})"
    )
    print(f"승인 한도: {decision_result.approved_amount_krw:,}원")

    report_md = build_rationale_report(decision_result)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"applicant_{applicant_id}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"근거 리포트 저장: {report_path}")

    payment_result = payment_mock.disburse_loan(
        applicant_id=applicant_id,
        decision=decision_result.final_decision,
        requested_loan_krw=decision_result.approved_amount_krw,
        rationale=decision_result.decision_reasoning,
    )
    print(f"집행 상태: {payment_result.status}")
    if payment_result.tx_signature:
        print(f"tx_signature: {payment_result.tx_signature}")
        print(f"explorer: {payment_result.explorer_url}")


def main():
    applicant_ids = list(SAMPLE_BUSINESS_DESCRIPTIONS.keys())
    for i, aid in enumerate(applicant_ids):
        if i > 0:
            time.sleep(15)  # 분당 rate limit 여유
        run_one(aid)

    print(f"\n{'=' * 60}")
    print(f"전체 완료. 판정 로그: onchain/payments_log.json / 리포트: {REPORT_DIR}")
    print(f"MOCK_MODE = {payment_mock.MOCK_MODE} (False면 실제 devnet 집행)")


if __name__ == "__main__":
    main()
