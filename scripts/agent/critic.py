"""Critic Agent — 1차 판정을 정책 문서 기준으로 독립적으로 재검토(반박/승인)한다.

은행권에서 여러 심사역이 교차검증하는 것을 흉내낸 2차 검토 단계다. 1차 판정 에이전트
(decision.py)가 이미 참고한 정량/정성/정책 정보를 그대로 다시 넣어주고, "이 판정이
근거와 정책에 실제로 부합하는가"만 별도의 Gemini 호출로 독립 재검토시킨다. record_decision과
동일하게 강제 function-calling(mode=ANY)으로 critic_review 호출을 받는다 — 구현 비용이 낮은
패턴을 그대로 재사용했다.
"""

from google.genai import types

from gemini_client import DECISION_MODEL, get_client, with_rate_limit_retry

_DECISION_KR = {"approve": "승인", "conditional": "조건부승인", "reject": "거절"}


def critic_review(verdict: str, critique_reasoning: str, policy_violation: str = "") -> dict:
    """1차 심사 판정을 검토한 뒤 승인 또는 반박한다. 이 함수를 반드시 호출해 검토 결과를 제출하라.

    Args:
        verdict: "approve"(1차 판정에 동의) 또는 "reject"(반박 — 1차 판정이 정책/근거와 맞지 않음).
        critique_reasoning: 검토 의견 (2~4문장, 한국어). 왜 동의/반박하는지 근거를 밝힌다.
        policy_violation: 위반된 것으로 보이는 정책 조항 제목. 위반이 없으면 빈 문자열.
    """
    return {
        "verdict": verdict,
        "critique_reasoning": critique_reasoning,
        "policy_violation": policy_violation,
    }


CRITIC_PROMPT = """당신은 소상공인 소액대출 심사의 2차 검토를 담당하는 Critic Agent입니다.
1차 심사 에이전트가 아래와 같이 판정했습니다. 1차 판정 에이전트와는 독립적으로, 이 판정이
제시된 정량/정성 정보와 정책 조항에 비추어 타당한지 재검토하고, 반드시 critic_review 함수를
호출해 결과를 제출하세요.

## 1차 판정
- 최종 판정: {final_decision} ({final_decision_kr})
- 정량 등급: {quant_tier} (부도확률 {prob:.1%})
- 정성 조정 여부: {adjustment_applied} ({adjustment_direction})
- 1차 판정 근거: {decision_reasoning}

## 1차 판정이 참조한 정책 조항
{policy_text}

## 검토 원칙
- 정량 등급이 "reject"인데 정성 조정으로 "approve"나 "conditional"로 상향했다면 반드시 반박(reject)하라 (정책 3항 위반).
- 판정 근거가 제시된 정량/정성 정보와 모순되거나, 근거 없이 등급을 조정했다면 반박하라.
- 그 외에는 정책과 근거가 합리적으로 일치하는지만 판단하고, 문제가 없으면 승인(approve)하라.
"""


def review_decision(
    final_decision: str,
    quant_tier: str,
    default_probability: float,
    adjustment_applied: bool,
    adjustment_direction: str,
    decision_reasoning: str,
    policy_citations_text: str,
) -> dict:
    """1차 판정 결과를 Critic Agent에게 검토시키고 {verdict, critique_reasoning, policy_violation}을 반환한다."""
    prompt = CRITIC_PROMPT.format(
        final_decision=final_decision,
        final_decision_kr=_DECISION_KR.get(final_decision, final_decision),
        quant_tier=quant_tier,
        prob=default_probability,
        adjustment_applied=adjustment_applied,
        adjustment_direction=adjustment_direction,
        decision_reasoning=decision_reasoning,
        policy_text=policy_citations_text or "(참조된 정책 조항 없음)",
    )

    client = get_client()
    generate = with_rate_limit_retry(client.models.generate_content)
    resp = generate(
        model=DECISION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[critic_review],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=["critic_review"],
                )
            ),
        ),
    )

    call = resp.candidates[0].content.parts[0].function_call
    return dict(call.args)


# 대시보드 "Critic Agent 단독 테스트" 버튼과 CLI(__main__)가 공유하는 데모 시나리오.
# 1차 판정 에이전트를 거치지 않고, Critic이 정책 위반을 실제로 잡아내는지/정상 판정은
# 통과시키는지를 바로 확인하기 위한 용도다 (decision.py의 실제 판정 흐름과는 무관).
DEMO_SCENARIOS = {
    "violation": dict(
        final_decision="approve",
        quant_tier="reject",
        default_probability=0.42,
        adjustment_applied=True,
        adjustment_direction="upgrade",
        decision_reasoning="정량 등급은 reject였으나 매출이 최근 회복세라 approve로 상향했다.",
        policy_citations_text="[정책 3항]\n정량 등급이 reject인 경우 정성 조정으로 상향할 수 없다.",
    ),
    "clean": dict(
        final_decision="conditional",
        quant_tier="conditional",
        default_probability=0.55,
        adjustment_applied=False,
        adjustment_direction="none",
        decision_reasoning="정량 등급 conditional을 그대로 유지했다. 매출 추세가 불명확해 상향 조정 근거가 부족하다고 판단했다.",
        policy_citations_text="[정책 5항]\n리스크 등급별 사후 관리 — conditional 등급은 한도의 50%만 우선 집행하고 3개월 후 재심사한다.",
    ),
}


if __name__ == "__main__":
    result = review_decision(**DEMO_SCENARIOS["violation"])
    print(result)
