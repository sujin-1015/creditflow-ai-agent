"""단일 Agent vs Multi Agent 비교 실험 러너 (프로토콜: docs/single_vs_multi_agent_experiment.md).

같은 케이스·같은 1차 판정(first pass)에서 세 조건을 한 번에 얻는다 (짝지은 비교):
  A. 단일 Agent        — Decision Agent의 1차 판정만 (검토 없음)
  B. 단일 Agent+자기검토 — 같은 Agent가 같은 대화 안에서 자기 판정을 다시 점검하고 필요하면 수정
  C. Multi Agent       — 1차 판정 + 독립 Critic (현재 운영 구조; Critic이 반박하면 집행 보류)

make_final_decision()은 수정하지 않고 그대로 호출한다 (A·C는 여기서 나온다). B는 그 호출에서 모델이 쓴
대화 이력(automatic_function_calling_history)을 그대로 이어받아 같은 모델·같은 도구로 한 턴 더 진행한다.
골드·케이스·정책·Decision/Critic 로직은 건드리지 않는다. 평가 하네스(eval_decision_agent_v2.py)의
경로 재지정·동결 파일 검사·재개 로직을 재사용한다.

실행 (같은 명령을 반복하면 이어서 진행):
    cd scripts/agent
    python eval_arms.py --condition production --runs 1,2
    python eval_arms.py --condition prefix --runs 1,2
산출물: models/eval_arms_<condition>/{manifest.json, results.jsonl}
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

import eval_decision_agent_v2 as h  # noqa: E402

PREFIX_PROMPT_COMMIT = "87e869e^"  # 프롬프트 규칙 4개를 일반화해 추가하기 직전 시점의 Decision Agent 지침
INTER_CASE_DELAY_SEC = 10

SELF_REVIEW_PROMPT = """방금 당신이 record_decision_tool로 제출한 판정을 스스로 다시 검토하세요.
아래 검토 원칙으로 점검하고, 판정이 정책이나 근거와 맞지 않는다고 판단하면
record_decision_tool을 다시 정확히 한 번 호출해 수정한 판정을 제출하세요.
문제가 없으면 도구를 호출하지 말고 "유지"라고만 답하세요.

## 검토 원칙
- 정량 등급이 "reject"인데 정성 조정으로 "approve"나 "conditional"로 상향했다면 반드시 수정하라 (정책 1.2절 위반).
- 판정 근거가 제시된 정량/정성 정보와 모순되거나, 근거 없이 등급을 조정했다면 수정하라.
- 그 외에는 정책과 근거가 합리적으로 일치하는지만 판단하고, 문제가 없으면 유지하라.
"""

EXTRA_FROZEN = [
    "scripts/agent/eval_arms.py",
    "docs/single_vs_multi_agent_experiment.md",
]


def load_prefix_instruction() -> str:
    src = subprocess.run(
        ["git", "show", f"{PREFIX_PROMPT_COMMIT}:scripts/agent/decision.py"],
        cwd=h.BASE_DIR, capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "AGENT_INSTRUCTION" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("이전 시점 decision.py에서 AGENT_INSTRUCTION을 찾지 못했습니다.")


class CallRecorder:
    """공용 Gemini 클라이언트의 generate_content 호출을 전부 기록한다 (결과는 그대로 통과).
    make_final_decision 안에서는 Decision 호출 → Critic 호출 순서로 기록되고, 이어서 B(자기검토) 호출이 기록된다."""

    def __init__(self):
        self.calls: list[dict] = []
        self._installed = False

    def install(self):
        if self._installed:
            return
        import gemini_client  # noqa: PLC0415

        client = gemini_client.get_client()
        orig = client.models.generate_content
        rec = self

        def wrapped(*a, **kw):
            t0 = time.time()
            resp = orig(*a, **kw)
            usage = getattr(resp, "usage_metadata", None)
            rec.calls.append({
                "kw": kw, "resp": resp, "elapsed_sec": round(time.time() - t0, 2),
                "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
                "model_version": getattr(resp, "model_version", None),
            })
            return resp

        client.models.generate_content = wrapped
        self._installed = True


def _fc_args(content) -> list[dict]:
    """Content에서 record_decision_tool 호출 인자를 모두 꺼낸다."""
    out = []
    for part in (getattr(content, "parts", None) or []):
        fc = getattr(part, "function_call", None)
        if fc is not None and fc.name == "record_decision_tool":
            out.append(dict(fc.args))
    return out


def run_self_review(rec: CallRecorder, decision_call: dict, first_pass: dict) -> dict:
    """B: 1차 판정을 만든 같은 대화(같은 모델·같은 도구) 안에서 자기검토 한 턴을 더 진행한다."""
    from google.genai import types  # noqa: PLC0415

    import gemini_client  # noqa: PLC0415

    resp1, kw1 = decision_call["resp"], decision_call["kw"]
    history = list(getattr(resp1, "automatic_function_calling_history", None) or [])
    if not history or history[0].role != "user":
        history = [types.Content(role="user", parts=[types.Part(text=kw1["contents"])])] + history
    final_model_content = resp1.candidates[0].content if resp1.candidates else None
    if final_model_content is not None and (not history or history[-1] is not final_model_content):
        history.append(final_model_content)
    history.append(types.Content(role="user", parts=[types.Part(text=SELF_REVIEW_PROMPT)]))

    client = gemini_client.get_client()
    generate = gemini_client.with_rate_limit_retry(client.models.generate_content)
    n_before = len(rec.calls)
    resp2 = generate(model=kw1["model"], contents=history, config=kw1["config"])
    call2 = rec.calls[n_before] if len(rec.calls) > n_before else {}

    new_entries = list(getattr(resp2, "automatic_function_calling_history", None) or [])[len(history):]
    revisions = [a for c in new_entries for a in _fc_args(c)]
    other_tools = [
        p.function_call.name for c in new_entries for p in (getattr(c, "parts", None) or [])
        if getattr(p, "function_call", None) is not None and p.function_call.name != "record_decision_tool"
    ]
    reply = ""
    try:
        reply = resp2.text or ""
    except Exception:  # noqa: BLE001
        pass

    last = revisions[-1] if revisions else None
    return {
        "called_record_decision": bool(revisions),
        "revised_decision": last["final_decision"] if last else None,
        "revised_adjustment_applied": last.get("adjustment_applied") if last else None,
        "revised_adjustment_direction": last.get("adjustment_direction") if last else None,
        "revised_reasoning": last.get("decision_reasoning") if last else None,
        "decision_after_self_review": last["final_decision"] if last else first_pass["final_decision"],
        "changed": bool(last) and last["final_decision"] != first_pass["final_decision"],
        "n_record_calls": len(revisions),
        "other_tools_called": other_tools,
        "reply_text": reply[:600],
        "elapsed_sec": call2.get("elapsed_sec"),
        "total_tokens": call2.get("total_tokens"),
    }


def run_arms_case(case: dict, legacy_fixtures: dict, rec: CallRecorder) -> dict:
    import copy  # noqa: PLC0415

    import decision  # noqa: PLC0415

    fixture = h.load_fixture_for(case, legacy_fixtures)
    decision.get_repayment_history = lambda applicant_id: copy.deepcopy(fixture)
    rec.install()
    rec.calls.clear()

    t0 = time.time()
    r = decision.make_final_decision(
        case["applicant_id"],
        is_reevaluation=bool(case["is_reevaluation"]),
        business_description_override=case["business_description"],
    )
    first_pass_sec = round(time.time() - t0, 1)
    first_pass = {
        "quant_tier": r.quant_tier,
        "default_probability": r.default_probability,
        "final_decision": r.final_decision,
        "adjustment_applied": r.adjustment_applied,
        "adjustment_direction": r.adjustment_direction,
        "decision_reasoning": r.decision_reasoning,
        "approved_amount_krw": r.approved_amount_krw,
        "policy_citations": r.policy_citations,
        "tool_call_summary": r.tool_call_summary,
    }
    critic = {
        "verdict": r.critic_verdict,
        "reasoning": r.critic_reasoning,
        "policy_violation": r.critic_policy_violation,
    }
    # make_final_decision 안의 호출: [0]=Decision 루프, [1]=Critic
    decision_call = rec.calls[0]
    critic_call = rec.calls[1] if len(rec.calls) > 1 else {}
    cost = {
        "first_pass_total_sec": first_pass_sec,
        "decision_call_sec": decision_call["elapsed_sec"],
        "decision_call_tokens": decision_call["total_tokens"],
        "critic_call_sec": critic_call.get("elapsed_sec"),
        "critic_call_tokens": critic_call.get("total_tokens"),
    }
    self_review = run_self_review(rec, decision_call, first_pass)
    return {
        "first_pass": first_pass,
        "critic": critic,
        "self_review": self_review,
        "cost": cost,
        "model_versions": sorted({c["model_version"] for c in rec.calls if c.get("model_version")}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["production", "prefix"], required=True,
                    help="production=현재 운영 Decision 지침, prefix=규칙 4개 추가 이전(스트레스 조건)")
    ap.add_argument("--runs", default="1")
    ap.add_argument("--max-cases", type=int, default=None, help="디버그용: 이번 실행에서 처리할 최대 케이스 수")
    ap.add_argument("--only", default=None, help="디버그용: 쉼표로 구분한 case_id만 실행")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--delay", type=int, default=INTER_CASE_DELAY_SEC)
    args = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_dir = args.out_dir or str(h.BASE_DIR / "models" / f"eval_arms_{args.condition}")
    h.configure(None, out_dir)
    h.FROZEN_FILES = h.FROZEN_FILES + [f for f in EXTRA_FROZEN if f not in h.FROZEN_FILES]

    import decision  # noqa: PLC0415

    prompt_meta = {"condition": args.condition}
    if args.condition == "prefix":
        decision.AGENT_INSTRUCTION = load_prefix_instruction()
        prompt_meta["decision_instruction_source"] = f"git {PREFIX_PROMPT_COMMIT}:scripts/agent/decision.py"
    else:
        prompt_meta["decision_instruction_source"] = "현재 scripts/agent/decision.py"
    prompt_meta["decision_instruction_sha256"] = hashlib.sha256(decision.AGENT_INSTRUCTION.encode("utf-8")).hexdigest()
    prompt_meta["self_review_prompt_sha256"] = hashlib.sha256(SELF_REVIEW_PROMPT.encode("utf-8")).hexdigest()

    data = json.loads(h.CASES_PATH.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"]]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["case_id"] in wanted]
    legacy_fixtures = json.loads(h.LEGACY_FIXTURE_PATH.read_text(encoding="utf-8"))
    manifest = h.check_manifest(allow_create=True)
    mpath = h.MANIFEST_PATH
    m = json.loads(mpath.read_text(encoding="utf-8"))
    if "arms_prompt" not in m:
        m["arms_prompt"] = prompt_meta
        mpath.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    elif m["arms_prompt"] != prompt_meta:
        raise SystemExit("이전 실행과 프롬프트 조건이 달라 중단합니다 (manifest.arms_prompt 불일치).")
    print(f"manifest OK (git_head_at_first_run={manifest['git_head_at_first_run'][:8]}) condition={args.condition}")

    runs = [int(x) for x in args.runs.split(",")]
    entries = h.load_entries()
    rec = CallRecorder()
    processed = 0
    first_api_call = True
    for run in runs:
        for idx, case in enumerate(cases, start=1):
            done, attempts = h.case_state(entries, "arms", run, case["case_id"])
            if done:
                continue
            if attempts >= h.MAX_TOTAL_ATTEMPTS:
                print(f"[run{run} {idx}/{len(cases)}] {case['case_id']}: 누적 시도 {attempts}회 초과 — 오류로 확정")
                continue
            if args.max_cases is not None and processed >= args.max_cases:
                print("max-cases 도달 — 종료")
                return 0
            for _ in range(h.MAX_ATTEMPTS_PER_INVOCATION):
                if attempts >= h.MAX_TOTAL_ATTEMPTS:
                    break
                if not first_api_call:
                    time.sleep(args.delay)
                first_api_call = False
                attempt_no = attempts + 1
                print(f"[run{run} {idx}/{len(cases)}] {case['case_id']} (신청자 {case['applicant_id']}, "
                      f"{case['category']}, 시도 {attempt_no})", flush=True)
                t0 = time.time()
                started = datetime.now(timezone.utc).isoformat()
                try:
                    result = run_arms_case(case, legacy_fixtures, rec)
                    entry = {"kind": "arms", "run": run, "case_id": case["case_id"], "attempt": attempt_no,
                             "status": "ok", "started_at": started, "elapsed_sec": round(time.time() - t0, 1),
                             "result": result}
                except Exception as e:  # noqa: BLE001
                    entry = {"kind": "arms", "run": run, "case_id": case["case_id"], "attempt": attempt_no,
                             "status": "error", "started_at": started, "elapsed_sec": round(time.time() - t0, 1),
                             "error": f"{type(e).__name__}: {e}"[:2000], "error_kind": h.classify_error(e),
                             "traceback_tail": traceback.format_exc()[-1500:]}
                h.append_entry(entry)
                entries.append(entry)
                if entry["status"] == "ok":
                    r = entry["result"]
                    sr = r["self_review"]
                    print(f"    -> 1차 {r['first_pass']['final_decision']} | Critic {r['critic']['verdict']} | "
                          f"자기검토 {'수정→' + str(sr['revised_decision']) if sr['changed'] else '유지'} "
                          f"({entry['elapsed_sec']}s)")
                    processed += 1
                    break
                print(f"    !! 오류[{entry['error_kind']}]: {entry['error'][:200]}")
                if entry["error_kind"] == "quota_daily":
                    print("일일 할당량 소진 — 기록 후 종료합니다. 같은 명령으로 이어서 재개하세요.")
                    return h.EXIT_QUOTA
                attempts += 1
                processed += 1 if attempts >= h.MAX_TOTAL_ATTEMPTS else 0
    print("요청한 회차의 처리가 끝났습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
