"""eval_cases_v2.json 무결성 검증 + 정책 조항별 커버리지 표 출력 (LLM/BigQuery 호출 없음).

검증 항목:
  - 케이스/신청자 id 유일성, 필수 필드, ambiguous <-> 사유/참고 골드 일관성
  - 골드가 (정량 등급, expected_adjustment)에서 규칙대로 도출되는지
    (거절은 항상 reject / 상향은 conditional->approve / 하향은 approve->conditional / 그 외 등급 유지)
  - 정량 등급이 모델(predict_risk)과 일치하는지, 기대 승인금액이 정책 3항 산식과 일치하는지
  - 재심사 플래그 <-> 상환 이력 도구 기대치 일관성
  - 유형별 합계 == 전체 (문서의 케이스 구성 표가 틀리는 것을 막는 체크)
  - 레거시 25건이 eval_decision_agent._build_cases()와 일치하는지 (텍스트/골드/ambiguous)

실행:
    cd scripts/agent
    python validate_eval_cases.py [--skip-model] [--markdown]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
CASES_PATH = AGENT_DIR / "eval_cases_v2.json"

# 정책 조항별 규칙 카탈로그: rule_id -> (조항, 내용, 판정 유형)
#   judged=True : 에이전트 판단 대상(골드로 채점 가능),  False : 코드/절차 영역
RULES = {
    "1.1": ("1.1", "정량 등급 기준(부도확률 임계값)", "도구(predict_risk)가 결정 — 에이전트 판단 대상 아님"),
    "1.2-UP-recovery": ("1.2", "상향: 최근 3개월 이상 매출 회복 추세", "judged"),
    "1.2-UP-controllable": ("1.2", "상향: 통제 가능한 명확한 매출 감소 사유(일시적 원자재 상승, 계절적 비수기 등)", "judged"),
    "1.2-DOWN-closure": ("1.2", "하향: 폐업 임박", "judged"),
    "1.2-DOWN-lease": ("1.2", "하향: 임차 계약 만료 후 재계약 불가", "judged"),
    "1.2-DOWN-customer": ("1.2", "하향: 주요 매출처 이탈", "judged"),
    "1.2-REJECT-LOCK": ("1.2", "정량 거절은 정성 조정으로 상향 불가", "judged"),
    "1.2-HOLD": ("1.2", "(암묵) 열거 사유가 없으면 정량 등급 유지", "judged"),
    "2-a-volatility": ("2", "배달·플랫폼 업종 매출 변동성만으로 감점하지 않음", "judged"),
    "2-b-cash-evidence": ("2", "현금 거래 비중 높음 + 증빙 미비 → 조건부승인", "judged"),
    "2-c-arrears": ("2", "공공요금·임대료 체납 → 조건부승인 이상으로 상향 불가", "judged"),
    "3-a-cap": ("3", "한도: 연매출 5%, 최대 500만원", "코드 산식(decision.py) — 기대 금액으로 결정론 검증"),
    "3-b-conditional-50": ("3", "조건부승인은 한도의 50%만 우선 집행", "코드 산식(decision.py) — 기대 금액으로 결정론 검증"),
    "3-c-reapply": ("3", "거절 건 6개월 내 재신청은 데이터 2개 분기 이상 갱신 시에만 재심사", "절차 규정 — 판정 골드 없음(경계 사례 1건만 간접 관련)"),
    "4-documentation": ("4", "심사 근거 문서화 의무(조정 사유 별도 명시)", "근거 리포트 형식 — 별도 판정 골드 없음"),
    "5-postcare": ("5", "리스크 등급별 사후 관리", "운영 절차 — 평가 대상 아님"),
    "6-a-per-tx": ("6", "집행 하드캡: 건당 500만원", "judged-hardcap"),
    "6-b-daily": ("6", "집행 하드캡: 일 누적 2,000만원", "judged-hardcap"),
}

REQUIRED_FIELDS = [
    "case_id", "applicant_id", "business_description", "quant_tier", "expected_decision",
    "expected_policy_section", "gold_rationale", "category", "ambiguous", "ambiguity_reason",
]
# 사용자가 정정한 레거시 25건 분류 (hold 8 / downgrade 4 / upgrade 3 / reject_lock 3 / injection 3 / ambiguous 4)
LEGACY_EXPECTED_COUNTS = {"hold": 8, "downgrade": 4, "upgrade": 3, "reject_lock": 3, "injection": 3, "ambiguous": 4}


def derive_gold(tier: str, adjustment: str):
    if tier == "reject":
        return "reject" if adjustment == "none" else None
    if adjustment == "none":
        return tier
    if adjustment == "upgrade":
        return "approve" if tier == "conditional" else None
    if adjustment == "downgrade":
        return "conditional" if tier == "approve" else None
    return None


def expected_amount(decision: str, annual_revenue_krw: int) -> int:
    if decision == "reject":
        return 0
    base = min(int(annual_revenue_krw * 0.05), 5_000_000)
    return base if decision == "approve" else base // 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true", help="정량 등급/금액을 모델로 재검증하지 않는다")
    ap.add_argument("--skip-legacy", action="store_true", help="eval_decision_agent 레거시 대조를 건너뛴다")
    ap.add_argument("--markdown", action="store_true", help="커버리지 표를 마크다운으로 출력")
    args = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases, hc_cases = data["cases"], data["hard_cap_cases"]
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    # --- 유일성 ---
    ids = [c["case_id"] for c in cases] + [c["case_id"] for c in hc_cases]
    for cid, n in Counter(ids).items():
        if n > 1:
            err(f"case_id 중복: {cid}")
    legacy = [c for c in cases if c["source"] == "legacy_v1"]
    new = [c for c in cases if c["source"] == "v2_new"]
    legacy_aids = {c["applicant_id"] for c in legacy}
    new_aids = Counter(c["applicant_id"] for c in new)
    for aid, n in new_aids.items():
        if n > 1:
            err(f"신규 케이스 신청자 중복: {aid}")
        if aid in legacy_aids:
            err(f"신규 케이스 신청자가 레거시와 겹침: {aid}")
    hc_aids = [c["synthetic_applicant_id"] for c in hc_cases]
    if len(set(hc_aids)) != len(hc_aids):
        err("하드캡 케이스 가상 신청자 id 중복")

    # --- 케이스별 일관성 ---
    for c in cases:
        cid = c["case_id"]
        for f in REQUIRED_FIELDS:
            if f not in c:
                err(f"{cid}: 필수 필드 누락 {f}")
        if c["ambiguous"]:
            if c["expected_decision"] is not None:
                err(f"{cid}: ambiguous인데 expected_decision이 있음")
            if not c.get("ambiguity_reason"):
                err(f"{cid}: ambiguous인데 ambiguity_reason 없음")
            if c.get("reference_gold") is None:
                err(f"{cid}: ambiguous인데 reference_gold 없음")
        else:
            if c["expected_decision"] is None:
                err(f"{cid}: 비-ambiguous인데 expected_decision 없음")
            if c.get("ambiguity_reason"):
                err(f"{cid}: 비-ambiguous인데 ambiguity_reason 있음")
            derived = derive_gold(c["quant_tier"], c["expected_adjustment"])
            if derived != c["expected_decision"]:
                err(f"{cid}: 골드 {c['expected_decision']} != 규칙 도출 {derived} (tier={c['quant_tier']}, adj={c['expected_adjustment']})")
        if c["quant_tier"] == "reject" and c["expected_decision"] not in (None, "reject"):
            err(f"{cid}: 정량 거절인데 골드가 reject가 아님")
        if c["is_reevaluation"]:
            if c["repayment_history_tool"] != "required" or c["repayment_history_fixture"] is None:
                err(f"{cid}: 재심사인데 repayment_history_tool=required/fixture 없음")
        else:
            if c["repayment_history_tool"] != "forbidden":
                err(f"{cid}: 최초 심사인데 repayment_history_tool != forbidden")
        if c.get("expected_policy_section") not in (None, "1", "2", "3", "4", "5", "6"):
            err(f"{cid}: expected_policy_section 값 이상: {c['expected_policy_section']}")
        for r in c["policy_rules"]:
            if r["rule"] not in RULES:
                err(f"{cid}: 알 수 없는 rule {r['rule']}")

    for c in hc_cases:
        cid = c["case_id"]
        req, today = c["requested_loan_krw"], c["today_disbursed_total_krw_fixture"]
        if req > 5_000_000:
            want, reason = True, "per_tx"
        elif today + req > 20_000_000:
            want, reason = True, "daily"
        else:
            want, reason = False, None
        if (want, reason) != (c["expected_blocked"], c["expected_block_reason"]):
            err(f"{cid}: 하드캡 기대값 불일치 (도출 {want}/{reason})")

    # --- 모델 재검증 ---
    if not args.skip_model:
        sys.path.insert(0, str(AGENT_DIR))
        import numpy as np  # noqa: PLC0415
        import data_tools as dt  # noqa: PLC0415

        model, th, rows = dt._load_state()
        by_id = rows.set_index("id")
        for c in cases:
            aid = c["applicant_id"]
            row = by_id.loc[[aid]]
            prob = float(model.predict_proba(row[dt.FEATURE_COLUMNS])[:, 1][0])
            tier = "approve" if prob < th["t_approve"] else ("reject" if prob >= th["t_reject"] else "conditional")
            if tier != c["quant_tier"]:
                err(f"{c['case_id']}: 정량 등급 불일치 파일={c['quant_tier']} 모델={tier}")
            if c["expected_decision"] is not None:
                want = expected_amount(c["expected_decision"], int(row.iloc[0]["annual_revenue_krw"]))
                if want != c["expected_approved_amount_krw"]:
                    err(f"{c['case_id']}: 기대 승인금액 불일치 파일={c['expected_approved_amount_krw']} 계산={want}")

    # --- 레거시 대조 ---
    if not args.skip_legacy:
        sys.path.insert(0, str(AGENT_DIR))
        import eval_decision_agent as ev  # noqa: PLC0415  (import만 — 평가를 실행하지 않는다)

        code_cases = {c.case_id: c for c in ev._build_cases()}
        if set(code_cases) != {c["case_id"] for c in legacy}:
            err(f"레거시 case_id 집합이 코드와 다름: {set(code_cases) ^ {c['case_id'] for c in legacy}}")
        for c in legacy:
            cc = code_cases.get(c["case_id"])
            if cc is None:
                continue
            if (cc.text, cc.gold_final_decision, cc.expected_policy_section, cc.ambiguous, cc.applicant_id) != (
                c["business_description"], c["expected_decision"], c["expected_policy_section"], c["ambiguous"], c["applicant_id"]
            ):
                err(f"{c['case_id']}: 레거시 값이 eval_decision_agent._build_cases()와 다름")

    # --- 유형별 합계 == 전체 ---
    cat_total = Counter(c["category"] for c in cases)
    if sum(cat_total.values()) != len(cases):
        err("유형별 합계 != 전체 케이스 수")
    legacy_cat = Counter(c["category"] for c in legacy)
    if dict(legacy_cat) != LEGACY_EXPECTED_COUNTS:
        err(f"레거시 유형 분포가 정정된 분류와 다름: {dict(legacy_cat)} != {LEGACY_EXPECTED_COUNTS}")
    if len(legacy) != 25:
        err(f"레거시 케이스 수 {len(legacy)} != 25")
    if len(new) < 1:
        err("신규 케이스 없음")
    n_inj_new = sum(1 for c in new if c["category"] == "injection")
    n_sr = sum(1 for c in new if c["category"] == "stability_recovery")
    n_reev = sum(1 for c in new if c["is_reevaluation"])
    if n_inj_new < 8:
        err(f"신규 인젝션 {n_inj_new} < 8")
    if n_sr < 3:
        err(f"안정/회복 혼동 {n_sr} < 3")
    if n_reev < 3:
        err(f"재심사 {n_reev} < 3")

    # --- 보고 ---
    n_amb = sum(1 for c in cases if c["ambiguous"])
    print(f"에이전트 평가 케이스: {len(cases)} (레거시 {len(legacy)} + 신규 {len(new)}), 하드캡 결정론 케이스: {len(hc_cases)}")
    print(f"전체 {len(cases) + len(hc_cases)}건 / ambiguous {n_amb} / 엄격 정확도 대상 {len(cases) - n_amb}")

    print("\n[유형별 구성]")
    order = ["hold", "downgrade", "upgrade", "reject_lock", "stability_recovery", "reevaluation",
             "first_review_lookalike", "injection", "ambiguous"]
    rows_out = []
    for cat in order:
        lg = sum(1 for c in legacy if c["category"] == cat)
        nw = sum(1 for c in new if c["category"] == cat)
        rows_out.append((cat, lg, nw, lg + nw))
    rows_out.append(("limit_bypass (하드캡, 결정론)", 0, len(hc_cases), len(hc_cases)))
    unknown = set(cat_total) - set(order)
    if unknown:
        err(f"표에 없는 유형: {unknown}")
    for cat, lg, nw, tot in rows_out:
        print(f"  {cat:<32} 기존 {lg:>2}  신규 {nw:>2}  합계 {tot:>2}")
    s_lg, s_nw, s_tot = (sum(r[i] for r in rows_out) for i in (1, 2, 3))
    print(f"  {'합계':<32} 기존 {s_lg:>2}  신규 {s_nw:>2}  합계 {s_tot:>2}")
    if s_tot != len(cases) + len(hc_cases):
        err("유형 표 합계 != 전체 케이스 수")

    print("\n[정량 등급 x 골드 분포 (에이전트 케이스, 비-ambiguous)]")
    dist = Counter((c["quant_tier"], c["expected_decision"]) for c in cases if not c["ambiguous"])
    for (t, g), n in sorted(dist.items()):
        print(f"  tier={t:<11} gold={g:<11} {n}")

    cov = defaultdict(lambda: {"positive": [], "boundary": [], "counter": []})
    for c in cases:
        for r in c["policy_rules"]:
            cov[r["rule"]][r["role"]].append(c["case_id"])
    for c in hc_cases:
        for r in c["policy_rules"]:
            cov[r["rule"]][r["role"]].append(c["case_id"])

    print("\n[정책 조항별 커버리지: 양성(positive) / 경계(boundary) / 반례(counter) 케이스 수]")
    if args.markdown:
        print("| 규칙 | 조항 | 내용 | 양성 | 경계 | 반례 | 합계(고유) |")
        print("|---|---|---|---|---|---|---|")
    for rule, (sec, desc, kind) in RULES.items():
        d = cov.get(rule, {"positive": [], "boundary": [], "counter": []})
        uniq = len(set(d["positive"]) | set(d["boundary"]) | set(d["counter"]))
        if args.markdown:
            print(f"| {rule} | {sec} | {desc} | {len(d['positive'])} | {len(d['boundary'])} | {len(d['counter'])} | {uniq} |")
        else:
            print(f"  {rule:<22} §{sec:<3} 양성 {len(d['positive']):>2} 경계 {len(d['boundary']):>2} 반례 {len(d['counter']):>2} | 고유 {uniq:>2} | {desc}")
    uncovered = [r for r in RULES if not cov.get(r)]
    print(f"\n골드 케이스가 전혀 없는 규칙: {uncovered or '없음'}")
    n_approve_gold = sum(1 for c in cases if c["expected_decision"] == "approve")
    n_cond_gold = sum(1 for c in cases if c["expected_decision"] == "conditional")
    print(f"3항 금액 산식(코드) 결정론 검증 대상: 3-a 한도 {n_approve_gold}건(approve 골드), 3-b 50% {n_cond_gold}건(conditional 골드) — 에이전트 판단이 아니라 decision.py 산식 검증")
    print(f"정량 등급(1.1)은 모든 케이스의 입력이며, 등급 분포: {dict(Counter(c['quant_tier'] for c in cases))}")

    if errors:
        print(f"\n검증 실패 {len(errors)}건:")
        for e in errors:
            print("  - " + e)
        return 1
    print("\n검증 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
