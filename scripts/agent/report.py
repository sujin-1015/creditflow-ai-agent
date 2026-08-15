"""판정 근거 자연어 리포트 생성.

decision.py의 FinalDecisionResult(정량 예측 + 정성 요약 + 정책 인용 + Gemini 판정 근거)를
사람이 읽기 좋은 마크다운 심사 리포트로 조립한다. 별도의 추가 Gemini 호출 없이,
이미 확보한 구조화 데이터를 템플릿으로 정리한다 (불필요한 API 호출/비용 절감).
"""

from datetime import datetime, timezone

from decision import FinalDecisionResult

DECISION_LABEL = {"approve": "승인", "conditional": "조건부승인", "reject": "거절"}
ADJUSTMENT_LABEL = {"upgrade": "상향 조정", "downgrade": "하향 조정", "none": "조정 없음"}
CRITIC_LABEL = {"approve": "승인 (1차 판정에 동의)", "reject": "반박 (재검토 필요)"}
_SOURCE_LABEL = {"fallback": "안전장치", "critic_agent": "Critic"}


def _format_tool_call_log(tool_call_log: list) -> str:
    if not tool_call_log:
        return "- (기록 없음)"
    lines = []
    for i, entry in enumerate(tool_call_log, start=1):
        tag = _SOURCE_LABEL.get(entry.get("source"))
        name = f"{entry['tool']}({tag})" if tag else entry["tool"]
        lines.append(f"{i}. `{name}` — {entry.get('args') or '(인자 없음)'}")
    return "\n".join(lines)


def build_rationale_report(result: FinalDecisionResult) -> str:
    biz = result.business_summary
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    adjustment_note = ""
    if result.adjustment_applied:
        adjustment_note = (
            f"\n> ⚠ 정량 등급({result.quant_tier})에서 정성 정보를 반영해 "
            f"**{ADJUSTMENT_LABEL[result.adjustment_direction]}**되었습니다.\n"
        )

    citations = "\n".join(f"- {c}" for c in result.policy_citations) or "- (해당 없음)"
    positive = "\n".join(f"- {s}" for s in biz["positive_signals"]) or "- (없음)"
    risk = "\n".join(f"- {s}" for s in biz["risk_signals"]) or "- (없음)"

    return f"""# 대출 심사 판정 리포트

- 신청자 ID: **{result.applicant_id}**
- 생성 시각: {now}

## 최종 판정: {DECISION_LABEL[result.final_decision]}
{adjustment_note}
- 승인 한도: **{result.approved_amount_krw:,}원**

## 1. 정량 분석
- 예측 부도확률: **{result.default_probability:.1%}**
- 모델 기준 등급: {DECISION_LABEL[result.quant_tier]}

## 2. 정성 분석 (사업자 설명 요약)
{biz['summary']}

- 매출 추세: {biz['revenue_trend']}
- 긍정 신호:
{positive}
- 위험 신호:
{risk}

## 3. 적용된 정책 조항
{citations}

## 4. 종합 판정 근거
{result.decision_reasoning}

## 5. Critic Agent 교차검증
- 검토 결과: {CRITIC_LABEL.get(result.critic_verdict, result.critic_verdict or "(검토 없음)")}
- 검토 의견: {result.critic_reasoning or "-"}
{f"- ⚠ 위반 의심 조항: {result.critic_policy_violation}" if result.critic_policy_violation else ""}

## 부록. 에이전트 도구 호출 로그
하드코딩된 순서가 아니라, 에이전트(Gemini)가 신청자 상황을 보고 스스로 판단해 아래 순서로
도구를 호출했다. `(안전장치)`가 붙은 항목은 모델이 호출하지 않아 코드가 직접 보강한 필수
정보, `(Critic)`은 1차 판정 이후 독립적으로 실행된 Critic Agent 호출이다.

{_format_tool_call_log(result.tool_call_log)}

---
*본 리포트는 PoC 시연용으로 자동 생성되었으며, 사업자 설명은 합성(synthetic) 텍스트를 사용했습니다.*
"""


if __name__ == "__main__":
    from decision import make_final_decision

    result = make_final_decision(142764)
    report = build_rationale_report(result)
    print(report)

    from pathlib import Path

    out_dir = Path(__file__).resolve().parent.parent.parent / "onchain" / "rationale_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"applicant_{result.applicant_id}.md"
    out_file.write_text(report, encoding="utf-8")
    print(f"\n저장 위치: {out_file}")
