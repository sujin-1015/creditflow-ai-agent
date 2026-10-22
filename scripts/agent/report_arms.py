"""단일 Agent vs Multi Agent 비교 실험 집계 (프로토콜: docs/single_vs_multi_agent_experiment.md).

models/eval_arms_<condition>/results.jsonl을 읽어 사전 정의한 지표를 계산한다. 프로토콜의 지표 정의를
바꾸지 않는다. 결과는 models/single_vs_multi_agent_report.json 과 표준출력으로 낸다.

실행:
    cd scripts/agent
    python report_arms.py
"""

import json
import sys
from pathlib import Path

import numpy as np

AGENT_DIR = Path(__file__).resolve().parent
BASE_DIR = AGENT_DIR.parent.parent
CASES_PATH = AGENT_DIR / "eval_cases_v2.json"
REPORT_PATH = BASE_DIR / "models" / "single_vs_multi_agent_report.json"
CONDITIONS = ("production", "prefix")
RANK = {"reject": 0, "conditional": 1, "approve": 2}
ARMS = ("A", "B", "C")
N_BOOT = 2000
SEED = 42


def load_instances(condition: str, gold: dict, ambiguous: set):
    path = BASE_DIR / "models" / f"eval_arms_{condition}" / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
    ok = {}
    errors = {}
    for r in rows:
        key = (r["run"], r["case_id"])
        if r["status"] == "ok":
            ok[key] = r["result"]
        else:
            errors[key] = errors.get(key, 0) + 1
    err_only = [k for k in errors if k not in ok]
    inst = []
    for (run, cid), res in sorted(ok.items()):
        if cid in ambiguous:
            continue
        g = gold[cid]
        d1 = res["first_pass"]["final_decision"]
        d2 = res["self_review"]["decision_after_self_review"]
        v = res["critic"]["verdict"]
        inst.append({
            "case": cid, "run": run, "gold": g, "d1": d1, "d2": d2, "critic": v, "res": res,
            "outcome": {
                "A": {"held": False, "dec": d1},
                "B": {"held": False, "dec": d2},
                "C": {"held": v != "approve", "dec": d1},  # Critic이 명시적으로 승인하지 않으면 보류 (payment_mock.should_hold_for_critic)
            },
        })
    return inst, {"n_ok_total": len(ok), "n_ambiguous_excluded": sum(1 for (_, c) in ok if c in ambiguous),
                  "n_error_instances": len(err_only), "error_instances": [list(k) for k in err_only]}


def arm_stats(inst: list[dict]) -> dict:
    n = len(inst)
    out = {"n": n}
    for arm in ARMS:
        unsafe = strict = correct_exec = held_block = held_false = executed = 0
        for x in inst:
            o = x["outcome"][arm]
            g = x["gold"]
            if o["held"]:
                if x["d1"] != g:
                    held_block += 1
                else:
                    held_false += 1
                continue
            executed += 1
            if o["dec"] == g:
                correct_exec += 1
            elif RANK[o["dec"]] > RANK[g]:
                unsafe += 1
            else:
                strict += 1
        wrong1 = [x for x in inst if x["d1"] != x["gold"]]
        unsafe1 = [x for x in inst if RANK[x["d1"]] > RANK[x["gold"]]]
        right1 = [x for x in inst if x["d1"] == x["gold"]]

        def caught(x):  # 1차 오류가 집행되지 않음/고쳐짐
            o = x["outcome"][arm]
            if arm == "A":
                return False
            if arm == "B":
                return o["dec"] == x["gold"]
            return o["held"]

        def unsafe_removed(x):  # 1차 과잉집행이 집행되지 않았거나 과잉이 아니게 고쳐짐
            o = x["outcome"][arm]
            if o["held"]:
                return True
            return RANK[o["dec"]] <= RANK[x["gold"]]

        def damaged(x):  # 1차가 맞았는데 이 조건에서 잘못되거나 보류됨
            o = x["outcome"][arm]
            return o["held"] if arm == "C" else (o["dec"] != x["gold"] if arm == "B" else False)

        out[arm] = {
            "unsafe_disbursement": unsafe, "unsafe_rate": unsafe / n if n else None,
            "strict_executed": strict, "held_error_blocked": held_block, "held_false_alarm": held_false,
            "executed": executed, "correct_and_executed": correct_exec,
            "correct_and_executed_rate": correct_exec / n if n else None,
            "exec_accuracy_among_executed": correct_exec / executed if executed else None,
            "first_pass_wrong_n": len(wrong1),
            "error_caught_rate": sum(caught(x) for x in wrong1) / len(wrong1) if wrong1 else None,
            "first_pass_unsafe_n": len(unsafe1),
            "unsafe_removed_rate": sum(unsafe_removed(x) for x in unsafe1) / len(unsafe1) if unsafe1 else None,
            "first_pass_correct_n": len(right1),
            "false_alarm_or_damage_rate": sum(damaged(x) for x in right1) / len(right1) if right1 else None,
        }
    return out


def bootstrap_diffs(inst: list[dict]) -> dict:
    """케이스 단위 부트스트랩(같은 케이스의 반복은 함께 재추출)으로 조건 간 차이의 95% 구간."""
    cases = sorted({x["case"] for x in inst})
    by_case = {c: [x for x in inst if x["case"] == c] for c in cases}
    rng = np.random.default_rng(SEED)
    keys = {"unsafe_rate": [], "correct_and_executed_rate": []}
    pairs = [("A", "C"), ("A", "B"), ("B", "C")]
    diffs = {f"{m}:{a}-{b}": [] for m in keys for a, b in pairs}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(cases), len(cases))
        sample = [x for i in pick for x in by_case[cases[i]]]
        st = arm_stats(sample)
        for m in keys:
            for a, b in pairs:
                diffs[f"{m}:{a}-{b}"].append(st[a][m] - st[b][m])
    return {
        k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
        for k, v in diffs.items()
    }


def cost_stats(inst: list[dict]) -> dict:
    def m(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    rs = [x["res"] for x in inst]
    return {
        "first_pass_sec_mean": m([r["cost"]["first_pass_total_sec"] for r in rs]),
        "first_pass_decision_call_tokens_mean": m([r["cost"]["decision_call_tokens"] for r in rs]),
        "critic_extra_calls": 1, "critic_extra_sec_mean": m([r["cost"]["critic_call_sec"] for r in rs]),
        "critic_extra_tokens_mean": m([r["cost"]["critic_call_tokens"] for r in rs]),
        "self_review_extra_calls_mean": m([1 + len(r["self_review"]["other_tools_called"]) for r in rs]),
        "self_review_extra_sec_mean": m([r["self_review"]["elapsed_sec"] for r in rs]),
        "self_review_extra_tokens_mean": m([r["self_review"]["total_tokens"] for r in rs]),
    }


def b_revision_detail(inst: list[dict]) -> dict:
    changed = [x for x in inst if x["res"]["self_review"]["changed"]]
    return {
        "n_changed": len(changed),
        "fixed (wrong→gold)": sum(1 for x in changed if x["d1"] != x["gold"] and x["d2"] == x["gold"]),
        "damaged (gold→wrong)": sum(1 for x in changed if x["d1"] == x["gold"] and x["d2"] != x["gold"]),
        "still_wrong": sum(1 for x in changed if x["d1"] != x["gold"] and x["d2"] != x["gold"]),
        "changes": [{"case": x["case"], "run": x["run"], "gold": x["gold"], "d1": x["d1"], "d2": x["d2"]} for x in changed],
    }


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    gold = {c["case_id"]: c["expected_decision"] for c in data["cases"]}
    ambiguous = {c["case_id"] for c in data["cases"] if c["ambiguous"]}

    report = {"protocol": "docs/single_vs_multi_agent_experiment.md", "conditions": {}}
    loaded = {}
    for cond in CONDITIONS:
        inst, meta = load_instances(cond, gold, ambiguous)
        loaded[cond] = inst
        if not inst:
            continue
        report["conditions"][cond] = {
            "meta": meta, "stats": arm_stats(inst), "bootstrap_diff": bootstrap_diffs(inst),
            "cost": cost_stats(inst), "self_review_changes": b_revision_detail(inst),
            "critic_on_first_pass": {
                "flagged_wrong": sum(1 for x in inst if x["d1"] != x["gold"] and x["critic"] == "reject"),
                "wrong_total": sum(1 for x in inst if x["d1"] != x["gold"]),
                "false_alarms": sum(1 for x in inst if x["d1"] == x["gold"] and x["critic"] == "reject"),
                "correct_total": sum(1 for x in inst if x["d1"] == x["gold"]),
            },
        }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for cond, rep in report["conditions"].items():
        st, meta = rep["stats"], rep["meta"]
        print(f"\n=== 조건: {cond} (인스턴스 {st['n']}건, 비-ambiguous; 오류로 결과 없음 {meta['n_error_instances']}건) ===")
        print(f"{'':<18}{'A 단일':>12}{'B 자기검토':>12}{'C Multi':>12}")
        rows = [
            ("과잉집행률(주지표)", "unsafe_rate"),
            ("맞게 집행된 비율", "correct_and_executed_rate"),
            ("오류 차단율", "error_caught_rate"),
            ("과잉집행 제거율", "unsafe_removed_rate"),
            ("오경보/훼손률", "false_alarm_or_damage_rate"),
        ]
        for label, key in rows:
            print(f"{label:<18}" + "".join(f"{pct(st[a][key]):>12}" for a in ARMS))
        print(f"{'과잉집행 건수':<18}" + "".join(f"{st[a]['unsafe_disbursement']:>12}" for a in ARMS)
              + f"   (1차 오답 {st['A']['first_pass_wrong_n']}건, 1차 과잉 {st['A']['first_pass_unsafe_n']}건)")
        print(f"{'보류(오류차단/오경보)':<14}" + f"{'-':>12}{'-':>12}"
              + f"{str(st['C']['held_error_blocked']) + '/' + str(st['C']['held_false_alarm']):>12}")
        bd = rep["bootstrap_diff"]
        for k in ("unsafe_rate:A-C", "unsafe_rate:A-B", "unsafe_rate:B-C", "correct_and_executed_rate:A-C"):
            d = bd[k]
            print(f"  {k}: {d['mean'] * 100:+.1f}%p (95% {d['ci95'][0] * 100:+.1f} ~ {d['ci95'][1] * 100:+.1f})")
        print("  B 수정:", {k: v for k, v in rep["self_review_changes"].items() if k != "changes"})
        print("  Critic(1차 기준):", rep["critic_on_first_pass"])
        c = rep["cost"]
        print(f"  비용: Critic +1호출 {c['critic_extra_sec_mean']}s/{c['critic_extra_tokens_mean']}토큰, "
              f"자기검토 +{c['self_review_extra_calls_mean']}호출 {c['self_review_extra_sec_mean']}s/{c['self_review_extra_tokens_mean']}토큰, "
              f"1차 판정 {c['first_pass_sec_mean']}s")
    print(f"\n저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
