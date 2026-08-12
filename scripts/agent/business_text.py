"""서류 텍스트(사업자 설명) 파싱/요약 로직.

원본 Kaggle 데이터셋에는 사업자등록증/매출설명 같은 비정형 텍스트가 없으므로,
PoC 시연을 위해 신청자 몇 명에 대한 가상의 사업자 설명문을 합성해두었다
(실 서비스라면 OCR/STT로 수집될 텍스트를 대신함 — SAMPLE_BUSINESS_DESCRIPTIONS 참고).

Gemini의 구조화 출력(JSON schema)으로 비정형 텍스트를 정형 요약으로 변환한다.
"""

from pydantic import BaseModel

from gemini_client import DECISION_MODEL, get_client, with_rate_limit_retry

# --- PoC용 합성 사업자 설명 (실 서비스에서는 OCR/STT로 수집될 텍스트) ---
SAMPLE_BUSINESS_DESCRIPTIONS: dict[int, str] = {
    10736: (
        "저는 7년째 같은 자리에서 인테리어 시공업을 하고 있습니다. 작년 하반기에 원자재(자재비) 가격이 "
        "일시적으로 급등하면서 3개월 정도 매출이 줄었는데, 최근 자재 가격이 안정되면서 지난달부터 "
        "신규 계약이 다시 늘고 있습니다. 거래처는 대부분 5년 이상 거래해온 단골 인테리어 대리점들입니다."
    ),
    192171: (
        "배달대행 라이더 중개 플랫폼을 11년째 운영 중입니다. 매출은 계절과 날씨에 따라 월별로 변동이 큰 편이고, "
        "이건 저희 업종 특성상 항상 있어온 자연스러운 변동입니다. 라이더 등록 수는 꾸준히 늘고 있고, "
        "제휴 음식점 수도 최근 3개월간 계속 증가하는 추세입니다."
    ),
    216524: (
        "전통시장에서 건어물 도소매업을 4년째 하고 있습니다. 최근 건물주가 임대 계약 갱신을 거부해서 "
        "다음 달까지 다른 곳으로 이전해야 하는 상황이고, 아직 마땅한 자리를 못 구했습니다. "
        "매출 증빙은 대부분 현금 거래라 카드 매출 자료만으로는 실제 매출의 절반 정도만 확인됩니다."
    ),
    142764: (
        "카페를 7년째 운영하고 있습니다. 지난 겨울 옆 건물 공사 때문에 3개월간 손님이 크게 줄어 매출이 "
        "일시적으로 떨어졌었는데, 공사가 끝난 지난달부터 매출이 공사 전 수준을 이미 회복했고 이번 달은 "
        "오히려 더 늘었습니다. 단골 손님 비중이 높고 임대 계약도 이번에 5년 재계약을 이미 마쳤습니다."
    ),
    61059: (
        "12년째 학원을 운영하고 있습니다. 매출 자체는 최근까지 꾸준한 편이었는데, 건물주가 재건축을 이유로 "
        "임대 계약 연장을 거부해서 다음 학기 전에 나가야 합니다. 아직 이전할 자리를 못 구했고, 주변 시세를 "
        "보니 지금보다 임대료가 훨씬 높아서 이전 자체가 사업적으로 가능할지도 확신이 서지 않는 상황입니다. "
        "가장 큰 매출을 차지하던 재수생 단과반 강사도 이번 달에 그만두면서 다른 학원으로 학생들을 많이 데려갔습니다."
    ),
}

# 대시보드 드롭다운 등 짧은 라벨이 필요한 곳에서 사용하는 업종 요약 (SAMPLE_BUSINESS_DESCRIPTIONS와 1:1 대응).
SAMPLE_BUSINESS_INDUSTRY: dict[int, str] = {
    10736: "인테리어 시공업",
    192171: "배달대행 라이더 중개 플랫폼",
    216524: "건어물 도소매업",
    142764: "카페 운영",
    61059: "학원 운영",
}

DEFAULT_DESCRIPTION_NOTE = (
    "(이 신청자에 대한 사업자 설명 텍스트가 없어 정성 정보 없이 정량 결과만으로 판단)"
)


class BusinessSummary(BaseModel):
    summary: str
    revenue_trend: str  # "증가" | "안정" | "감소" | "불명"
    positive_signals: list[str]
    risk_signals: list[str]


SUMMARIZE_PROMPT = """다음은 소상공인 대출 신청자가 작성한 사업 설명이다. 이를 분석해 아래 항목을 채워라.

- summary: 2~3문장 요약
- revenue_trend: "증가", "안정", "감소", "불명" 중 하나
- positive_signals: 상환능력에 긍정적인 신호 목록 (없으면 빈 리스트)
- risk_signals: 상환능력에 부정적인 신호 목록 (없으면 빈 리스트, 예: 폐업 임박, 임대계약 문제, 매출 감소 등)

사업 설명:
\"\"\"{text}\"\"\"
"""


@with_rate_limit_retry
def summarize_business_text(text: str) -> BusinessSummary:
    """비정형 사업자 설명 텍스트를 구조화된 요약으로 변환한다 (Gemini 구조화 출력)."""
    client = get_client()
    resp = client.models.generate_content(
        model=DECISION_MODEL,
        contents=SUMMARIZE_PROMPT.format(text=text),
        config={
            "response_mime_type": "application/json",
            "response_schema": BusinessSummary,
        },
    )
    return resp.parsed


def get_business_description(applicant_id: int) -> str:
    return SAMPLE_BUSINESS_DESCRIPTIONS.get(applicant_id, DEFAULT_DESCRIPTION_NOTE)


if __name__ == "__main__":
    for aid, text in SAMPLE_BUSINESS_DESCRIPTIONS.items():
        summary = summarize_business_text(text)
        print(f"\n=== 신청자 {aid} ===")
        print(summary.model_dump_json(indent=2, exclude_none=False))
