"""조건부승인 건의 3개월 후 자동 재심사 (정책 3항/5항 자동화).

정책 문서: "조건부승인 건은 한도의 50%까지만 우선 집행하고, 3개월 후 재심사를 통해
잔여 한도 집행 여부를 결정한다." — 지금까지는 문서상 규칙이었을 뿐 자동화가 안 되어
있었다. 이 모듈이 그 자동화를 담당한다.

실제 서비스라면 3개월 뒤 갱신된 매출 데이터/사업자 설명이 들어오겠지만, 이 PoC는
정적 데이터셋이라 "3개월 후 상황"을 나타내는 합성(synthetic) 후속 텍스트로 대체한다
(business_text.py의 최초 합성 텍스트와 동일한 성격 — 명시적으로 가상 시나리오).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import business_text  # noqa: E402
import bigquery_logger  # noqa: E402
import payment_mock  # noqa: E402
from decision import make_final_decision  # noqa: E402

# --- PoC용 합성 "3개월 후" 후속 상황 (실 서비스라면 갱신된 실제 데이터로 대체) ---
FOLLOWUP_BUSINESS_UPDATES: dict[int, str] = {
    61059: (
        "지난번 재계약 문제로 조건부승인을 받았던 학원입니다. 그 사이에 도보 5분 거리에 "
        "새 자리를 구해 무사히 이전을 마쳤고, 기존 수강생의 90% 이상이 새 위치로도 "
        "그대로 옮겨와 주었습니다. 이탈했던 핵심 강사 자리도 새로운 강사를 채용해 "
        "안정적으로 대체했고, 지난 두 달간 매출은 이전 수준을 회복했습니다."
    ),
}


def reevaluate(applicant_id: int, followup_text: str | None = None) -> dict:
    """조건부 건 하나를 재심사한다. 승인으로 상향되면 잔여 50%를 즉시 집행한다."""
    followup_text = followup_text or FOLLOWUP_BUSINESS_UPDATES.get(applicant_id)
    if followup_text is None:
        raise ValueError(f"applicant_id {applicant_id}에 대한 후속 텍스트가 없습니다.")

    # 재심사 시점의 최신 상황으로 사업자 설명을 교체 (원본 데모용 텍스트는 다른 프로세스에 영향 없음)
    business_text.SAMPLE_BUSINESS_DESCRIPTIONS[applicant_id] = followup_text

    new_result = make_final_decision(applicant_id, is_reevaluation=True)
    upgraded = new_result.final_decision == "approve"

    reeval_record = {
        "applicant_id": applicant_id,
        "original_decision": "conditional",
        "new_decision": new_result.final_decision,
        "upgraded": upgraded,
        "additional_amount": 0.0,
        "currency": payment_mock.CURRENCY,
        "tx_signature": None,
        "explorer_url": None,
        "rationale": new_result.decision_reasoning,
        "rationale_hash": None,
        "critic_verdict": new_result.critic_verdict,
        "critic_reasoning": new_result.critic_reasoning,
        "tool_call_summary": new_result.tool_call_summary,
        "reevaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    if upgraded:
        wallet_address = _get_wallet(applicant_id)
        payment_rationale = f"[재심사 상향] {new_result.decision_reasoning}"
        payment_result = payment_mock.disburse_remaining(
            applicant_id=applicant_id,
            wallet_address=wallet_address,
            rationale=payment_rationale,
        )
        # 해시가 온체인 메모와 대조 가능하도록, 실제로 해시/서명한 문자열로 rationale을 맞춘다.
        reeval_record["rationale"] = payment_rationale
        reeval_record["rationale_hash"] = payment_result.rationale_hash
        reeval_record["additional_amount"] = payment_result.devnet_test_amount
        reeval_record["tx_signature"] = payment_result.tx_signature
        reeval_record["explorer_url"] = payment_result.explorer_url

    bigquery_logger.log_reevaluation(reeval_record)
    return reeval_record


def _get_wallet(applicant_id: int) -> str:
    import devnet_transfer

    return devnet_transfer.get_or_create_devnet_wallet(f"applicant_{applicant_id}")


def run_due_reevaluations(min_days: int = 90) -> list[dict]:
    """min_days 이상 지난 conditional 건을 전부 찾아 재심사한다 (Cloud Scheduler가 호출)."""
    due = bigquery_logger.find_due_conditionals(min_days=min_days)
    results = []
    for row in due:
        applicant_id = row["applicant_id"]
        if applicant_id not in FOLLOWUP_BUSINESS_UPDATES:
            continue  # PoC 시연용 후속 텍스트가 준비된 건만 처리
        results.append(reevaluate(applicant_id))
    return results


if __name__ == "__main__":
    result = reevaluate(61059)
    print("=== 재심사 결과 ===")
    print(f"신청자: {result['applicant_id']}")
    print(f"기존 판정: {result['original_decision']} -> 새 판정: {result['new_decision']}")
    print(f"상향 여부: {result['upgraded']}")
    if result["upgraded"]:
        print(f"잔여 집행액: {result['additional_amount']} {result['currency']}")
        print(f"tx: {result['tx_signature']}")
        print(f"explorer: {result['explorer_url']}")
    print(f"근거: {result['rationale']}")
