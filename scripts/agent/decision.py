"""최종 판정 로직 — Gemini function calling으로 정량 예측 + 정성 정보 + 정책(RAG)을 종합한다.

흐름:
    1. predict_risk(quant tool) -> 정량 등급/확률
    2. get_applicant_data(data tool) -> 신청자 정형 데이터
    3. summarize_business_text -> 사업자 설명 구조화 요약(정성 정보)
    4. PolicyRAG.retrieve -> 관련 정책 조항 검색
    5. Gemini function calling(mode=ANY)으로 record_decision 함수를 강제 호출시켜
       최종 판정(승인/조건부승인/거절), 조정 여부, 근거를 구조화된 형태로 받는다.
"""

from dataclasses import dataclass, field
from typing import Optional

from google.genai import types

from business_text import get_business_description, summarize_business_text
from data_tools import get_applicant_data, predict_risk
from gemini_client import DECISION_MODEL, get_client, with_rate_limit_retry
from rag import PolicyRAG

LOAN_RATE = 0.05
LOAN_CAP_KRW = 5_000_000

_rag = None


def _get_rag() -> PolicyRAG:
    global _rag
    if _rag is None:
        _rag = PolicyRAG()
    return _rag


@dataclass
class FinalDecisionResult:
    applicant_id: int
    quant_tier: str
    default_probability: float
    final_decision: str
    adjustment_applied: bool
    adjustment_direction: str
    decision_reasoning: str
    approved_amount_krw: int
    business_summary: dict
    policy_citations: list = field(default_factory=list)


def record_decision(
    final_decision: str,
    adjustment_applied: bool,
    adjustment_direction: str,
    decision_reasoning: str,
) -> dict:
    """최종 심사 판정을 기록한다. 이 함수를 반드시 호출해 결과를 제출하라.

    Args:
        final_decision: "approve", "conditional", "reject" 중 하나 (최종 판정).
        adjustment_applied: 정량 등급을 정성 정보로 상향/하향 조정했는지 여부.
        adjustment_direction: "upgrade", "downgrade", "none" 중 하나.
        decision_reasoning: 정량 확률, 정성 요약, 정책 조항을 종합한 한국어 판정 근거 (3~5문장).
    """
    return {
        "final_decision": final_decision,
        "adjustment_applied": adjustment_applied,
        "adjustment_direction": adjustment_direction,
        "decision_reasoning": decision_reasoning,
    }


DECISION_PROMPT = """당신은 소상공인 소액대출 심사 에이전트입니다. 아래 정량/정성/정책 정보를 종합해
최종 판정을 내리고 반드시 record_decision 함수를 호출해 결과를 제출하세요.

## 정량 예측 결과
- 예측 부도확률: {prob:.1%}
- 정량 등급(모델 임계값 기준): {quant_tier}

## 사업자 설명 요약 (정성 정보)
- 요약: {summary}
- 매출 추세: {revenue_trend}
- 긍정 신호: {positive_signals}
- 위험 신호: {risk_signals}

## 관련 정책 조항 (RAG 검색 결과)
{policy_text}

## 판정 시 유의사항
- 정량 등급이 "reject"이면 정성 조정으로 상향할 수 없습니다 (정책 3항).
- 정량 등급이 "conditional"인데 명확한 매출 회복/통제 가능한 감소 사유가 있으면 "approve"로 상향 검토하세요.
- 정량 등급이 "approve"라도 폐업 임박/임대 문제/주요 매출처 이탈 등 심각한 위험 신호가 있으면 "conditional"로 하향 검토하세요.
"""


def make_final_decision(applicant_id: int) -> FinalDecisionResult:
    quant = predict_risk(applicant_id)
    applicant = get_applicant_data(applicant_id)
    biz_text = get_business_description(applicant_id)
    biz_summary = summarize_business_text(biz_text)

    policy_query = (
        f"정량등급 {quant['quant_tier']}, 매출추세 {biz_summary.revenue_trend}, "
        f"긍정신호 {biz_summary.positive_signals}, 위험신호 {biz_summary.risk_signals}"
    )
    policy_hits = _get_rag().retrieve(policy_query, top_k=3)
    policy_text = "\n\n".join(f"[{h['title']}]\n{h['text']}" for h in policy_hits)

    prompt = DECISION_PROMPT.format(
        prob=quant["default_probability"],
        quant_tier=quant["quant_tier"],
        summary=biz_summary.summary,
        revenue_trend=biz_summary.revenue_trend,
        positive_signals=biz_summary.positive_signals,
        risk_signals=biz_summary.risk_signals,
        policy_text=policy_text,
    )

    client = get_client()
    generate = with_rate_limit_retry(client.models.generate_content)
    resp = generate(
        model=DECISION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[record_decision],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=["record_decision"],
                )
            ),
        ),
    )

    call = resp.candidates[0].content.parts[0].function_call
    args = dict(call.args)

    final_decision = args["final_decision"]
    loan_amount = 0
    if final_decision in ("approve", "conditional"):
        base_amount = min(int(applicant["annual_revenue_krw"] * LOAN_RATE), LOAN_CAP_KRW)
        loan_amount = base_amount if final_decision == "approve" else base_amount // 2

    return FinalDecisionResult(
        applicant_id=applicant_id,
        quant_tier=quant["quant_tier"],
        default_probability=quant["default_probability"],
        final_decision=final_decision,
        adjustment_applied=args["adjustment_applied"],
        adjustment_direction=args["adjustment_direction"],
        decision_reasoning=args["decision_reasoning"],
        approved_amount_krw=loan_amount,
        business_summary=biz_summary.model_dump(),
        policy_citations=[h["title"] for h in policy_hits],
    )


if __name__ == "__main__":
    import time

    for i, aid in enumerate((10736, 192171, 216524)):
        if i > 0:
            time.sleep(20)  # 무료 티어 분당 호출 한도 여유를 두기 위한 간격
        result = make_final_decision(aid)
        print(f"\n=== 신청자 {result.applicant_id} ===")
        print(f"정량 등급: {result.quant_tier} (부도확률 {result.default_probability:.1%})")
        print(f"최종 판정: {result.final_decision} (조정: {result.adjustment_applied}/{result.adjustment_direction})")
        print(f"승인 금액: {result.approved_amount_krw:,}원")
        print(f"근거: {result.decision_reasoning}")
        print(f"참조 정책: {result.policy_citations}")
