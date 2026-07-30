"""판정 근거 자연어 리포트 생성.

decision.py의 FinalDecisionResult(정량 예측 + 정성 요약 + 정책 인용 + Gemini 판정 근거)를
사람이 읽기 좋은 마크다운 심사 리포트로 조립한다. 별도의 추가 Gemini 호출 없이,
이미 확보한 구조화 데이터를 템플릿으로 정리한다 (불필요한 API 호출/비용 절감).
"""

from datetime import datetime, timezone

from decision import FinalDecisionResult

DECISION_LABEL = {"approve": "승인", "conditional": "조건부승인", "reject": "거절"}
ADJUSTMENT_LABEL = {"upgrade": "상향 조정", "downgrade": "하향 조정", "none": "조정 없음"}


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
