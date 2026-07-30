"""견고성 테스트 — '깔끔한' 시연용 텍스트가 아니라 실제 사업자가 쓸 법한 텍스트로 검증.

business_text.py의 SAMPLE_BUSINESS_DESCRIPTIONS 5개는 정책 문서 조항에 정확히 맞춰
제가 직접 쓴 시연용 텍스트라, "메커니즘이 작동한다"는 증명이지 "실제 사업자들의
애매하고 앞뒤 안 맞고 감정적인 텍스트에도 잘 작동한다"는 증명은 아니다.

이 스크립트는 아래 4가지 어려운 케이스로 그 부분을 테스트한다:
    1. 두서없고 사투리/구어체 섞인 횡설수설 텍스트 (신호 추출이 어려운 경우)
    2. 근거 없는 과장/허풍 (Gemini가 검증 안 된 낙관을 과신하지 않는지)
    3. 모순되고 애매한 서술 (좋다/나쁘다가 뒤섞여 명확한 결론이 안 나는 경우)
    4. 객관적 근거 없이 감정에 호소하는 텍스트 (동정心에 판정이 흔들리는지)

정량 등급이 이미 정해진(quant_tier) 4명에게 각각 붙여서, 판정이 여전히
"정량 우선 + 정책 근거 기반"으로 나오는지 관찰한다.
"""

import time

from decision import make_final_decision

NOISY_DESCRIPTIONS = {
    # 1. 횡설수설 (quant: approve, 84175, 부도확률 28.7%, 업력 2년)
    84175: (
        "저요 분식집 하는데예 진짜 요즘 힘든게 많아요 알바도 잘 안구해지고 재료값도 오르고 "
        "근데 뭐 그래도 단골들이 있어가지고 어떻게든 되긴 되는데 작년에 비해서는 좀 나은거 "
        "같기도 하고 아닌거 같기도 하고 잘 모르겠어요 아무튼 열심히 하고 있습니다 대출 되면 "
        "진짜 감사하겠습니다"
    ),
    # 2. 근거 없는 과장 (quant: approve, 124720, 부도확률 28.9%, 업력 13년)
    124720: (
        "저희 가게는 완전 대박날 예정입니다! 곧 유명 유튜버가 촬영하러 온다고 했고 SNS에서도 "
        "입소문 나고 있어서 다음 달부터는 매출이 지금의 몇 배는 뛸 것 같습니다. 13년 동안 "
        "운영하면서 이렇게 좋은 기회는 처음이에요. 무조건 잘 될 거라고 확신합니다."
    ),
    # 3. 모순/애매 (quant: conditional, 221661, 부도확률 59.3%, 업력 11년)
    221661: (
        "매출이요? 늘었다고 해야 하나... 사실 재작년보다는 조금 나은데 작년보다는 또 별로였고 "
        "이번달은 또 괜찮았어요. 대출금이 좀 있긴 한데 그건 예전부터 있던 거라 크게 문제는 "
        "안 될 것 같고, 그냥 이것저것 겹쳐서 좀 복잡하네요. 그래도 11년째 하고 있으니까 "
        "어떻게든 되지 않을까 싶습니다."
    ),
    # 4. 감정 호소 (quant: reject, 238412, 부도확률 73.8%, 업력 5년)
    238412: (
        "정말 힘든 시기를 보내고 있습니다. 최근에 건강도 안 좋아지고 가족 일도 겹쳐서 마음이 "
        "많이 힘든 상황이에요. 그래도 5년 동안 정말 열심히 해왔고 손님들도 저를 많이 "
        "응원해주십니다. 이번에 대출만 받을 수 있으면 정말 다시 일어설 수 있을 것 같아요. "
        "제발 도와주셨으면 좋겠습니다."
    ),
}

TEST_LABELS = {
    84175: "횡설수설/신호 추출 난이도",
    124720: "근거 없는 과장/허풍",
    221661: "모순되고 애매한 서술",
    238412: "객관적 근거 없는 감정 호소",
}


def main():
    import business_text

    for i, (applicant_id, text) in enumerate(NOISY_DESCRIPTIONS.items()):
        if i > 0:
            time.sleep(15)

        # 이 테스트 동안만 노이즈 텍스트로 교체 (SAMPLE_BUSINESS_DESCRIPTIONS 원본은 안 건드림)
        business_text.SAMPLE_BUSINESS_DESCRIPTIONS[applicant_id] = text

        print(f"\n{'=' * 70}")
        print(f"[테스트: {TEST_LABELS[applicant_id]}] 신청자 {applicant_id}")
        print(f"{'=' * 70}")
        print(f"입력 텍스트: {text[:80]}...")

        result = make_final_decision(applicant_id)
        print(f"\n정량 등급: {result.quant_tier} (부도확률 {result.default_probability:.1%})")
        print(f"정성 요약: {result.business_summary['summary']}")
        print(f"매출추세: {result.business_summary['revenue_trend']}")
        print(f"긍정신호: {result.business_summary['positive_signals']}")
        print(f"위험신호: {result.business_summary['risk_signals']}")
        print(f"\n최종 판정: {result.final_decision} (조정: {result.adjustment_applied}/{result.adjustment_direction})")
        print(f"근거: {result.decision_reasoning}")


if __name__ == "__main__":
    main()
