"""Decision Agent 평가 하네스 v2 — eval_cases_v2.json(사전 정의된 골드)을 `make_final_decision()` 전체
파이프라인으로 돌리고 케이스별 원시 결과를 즉시 JSONL에 기록한다. 에이전트 로직·프롬프트·정책 문서는
건드리지 않는다 (v1 `eval_decision_agent.py`는 그대로 두고, 이 파일이 v2 전용 러너다).

v1 대비 하네스 변경점 (에이전트 동작에는 영향 없음):
  - 재심사 케이스는 `is_reevaluation=True`로 호출한다 (운영 경로 reevaluation.reevaluate와 동일).
    최초 심사 케이스는 v1과 똑같이 `make_final_decision(aid, business_description_override=text)`.
  - `get_repayment_history`는 BigQuery 대신 고정값을 반환한다: 신규 케이스는 파일의 fixture,
    기존 25건은 `eval_fixtures_legacy_repayments.json`(v1 당시 BigQuery 값 스냅샷). 결과가 외부
    상태(테이블에 쌓이는 데모 상환 기록)에 따라 달라지지 않게 하기 위함.
  - 하드캡 케이스는 LLM 없이 `payment_mock._check_hard_caps`를 직접 호출하고, 일 누적 집행액만
    고정값으로 주입한다.
  - 케이스별로 즉시 기록(append + fsync)하고 재개할 수 있다. API 오류는 제외하지 않고 오류로
    기록한 뒤 재시도한다. 일일 할당량(PerDay) 소진 시에는 기록 후 종료(코드 75) — 다음 실행에서 이어서 한다.
  - 응답의 `model_version`을 기록한다 (모델 별칭 `gemini-flash-lite-latest`이 실제로 어느 버전이었는지).

실행 (같은 명령을 반복 실행하면 이어서 진행):
    cd scripts/agent
    python eval_decision_agent_v2.py --runs 1        # 주 결과(1회차)
    python eval_decision_agent_v2.py --runs 2,3      # 판정 안정성용 반복

산출물: models/eval_v2_runs/{manifest.json, results.jsonl}
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = AGENT_DIR.parent
BASE_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

CASES_PATH = AGENT_DIR / "eval_cases_v2.json"
LEGACY_FIXTURE_PATH = AGENT_DIR / "eval_fixtures_legacy_repayments.json"
RUN_DIR = BASE_DIR / "models" / "eval_v2_runs"
RESULTS_PATH = RUN_DIR / "results.jsonl"
MANIFEST_PATH = RUN_DIR / "manifest.json"

INTER_CASE_DELAY_SEC = 20  # v1과 동일 — 줄이지 않는다
MAX_ATTEMPTS_PER_INVOCATION = 3  # 한 번 실행에서 케이스당 최대 시도
MAX_TOTAL_ATTEMPTS = 6  # 일일 할당량 오류를 제외한 누적 시도 상한 (넘으면 오류로 확정)
EXIT_QUOTA = 75

# 실행 중 바뀌면 안 되는 파일들 (에이전트 로직·프롬프트·정책·골드)
FROZEN_FILES = [
    "scripts/agent/decision.py", "scripts/agent/critic.py", "scripts/agent/rag.py",
    "scripts/agent/business_text.py", "scripts/agent/gemini_client.py", "scripts/agent/data_tools.py",
    "scripts/agent/policy_docs/loan_policy.md", "scripts/agent/eval_cases_v2.json",
    "scripts/agent/eval_fixtures_legacy_repayments.json", "models/xgboost_model.joblib",
    "models/threshold_config.json",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def build_manifest() -> dict:
    dirty = _git("status", "--porcelain", "--", *FROZEN_FILES)
    return {
        "frozen_file_sha256": {f: _sha256(BASE_DIR / f) for f in FROZEN_FILES},
        "frozen_files_dirty_in_git": dirty or None,
        "git_head": _git("rev-parse", "HEAD"),
    }


def check_manifest(allow_create: bool) -> dict:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    current = build_manifest()
    if current["frozen_files_dirty_in_git"]:
        raise SystemExit(f"동결 파일에 커밋되지 않은 변경이 있어 중단합니다:\n{current['frozen_files_dirty_in_git']}")
    if not MANIFEST_PATH.exists():
        if not allow_create:
            raise SystemExit("manifest.json이 없습니다.")
        import decision  # noqa: PLC0415
        import gemini_client  # noqa: PLC0415

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frozen_file_sha256": current["frozen_file_sha256"],
            "git_head_at_first_run": current["git_head"],
            "config": {
                "decision_model_alias": gemini_client.DECISION_MODEL,
                "embedding_model": gemini_client.EMBEDDING_MODEL,
                "temperature": "미지정 (API 기본값; decision.py/critic.py가 GenerateContentConfig에 temperature를 주지 않음)",
                "seed": "미지정",
                "top_k_policy_retrieval": 3,
                "inter_case_delay_sec": INTER_CASE_DELAY_SEC,
                "planned_runs": 3,
                "primary_run": 1,
                "note": "1회차를 주 결과로 확정하고 2~3회차는 판정 안정성 지표로만 보고한다.",
            },
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["frozen_file_sha256"] != current["frozen_file_sha256"]:
        diff = [f for f in FROZEN_FILES if manifest["frozen_file_sha256"].get(f) != current["frozen_file_sha256"].get(f)]
        raise SystemExit(f"실행 도중 동결 파일이 바뀌어 중단합니다: {diff}")
    return manifest


# ---------------------------------------------------------------- 결과 기록/재개
def load_entries() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    out = []
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def append_entry(entry: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def case_state(entries: list[dict], kind: str, run: int, case_id: str) -> tuple[bool, int]:
    """(완료 여부, 일일 할당량 오류를 제외한 누적 시도 횟수)"""
    done, attempts = False, 0
    for e in entries:
        if e["kind"] == kind and e["run"] == run and e["case_id"] == case_id:
            if e["status"] == "ok":
                done = True
            elif e.get("error_kind") != "quota_daily":
                attempts += 1
    return done, attempts


# ---------------------------------------------------------------- 파이프라인 연결
def _parse_ts(rec: dict) -> dict:
    rec = dict(rec)
    ts = rec.get("timestamp")
    if isinstance(ts, str):
        rec["timestamp"] = datetime.fromisoformat(ts)  # BigQuery가 datetime을 돌려주던 것과 같은 타입
    return rec


def load_fixture_for(case: dict, legacy_fixtures: dict) -> list[dict]:
    if case.get("repayment_history_fixture") is not None:
        raw = case["repayment_history_fixture"]
    elif case["source"] == "legacy_v1":
        raw = legacy_fixtures["by_applicant"].get(str(case["applicant_id"]), [])
    else:
        raw = []
    return [_parse_ts(r) for r in copy.deepcopy(raw)]


class _VersionRecorder:
    """client.models.generate_content를 감싸 응답의 model_version을 기록한다 (호출 결과는 그대로 통과)."""

    def __init__(self):
        self.versions: list[str] = []
        self._installed = False

    def install(self):
        if self._installed:
            return
        import gemini_client  # noqa: PLC0415

        client = gemini_client.get_client()
        orig = client.models.generate_content
        rec = self

        def wrapped(*a, **kw):
            resp = orig(*a, **kw)
            mv = getattr(resp, "model_version", None)
            if mv:
                rec.versions.append(mv)
            return resp

        client.models.generate_content = wrapped
        self._installed = True


def run_agent_case(case: dict, legacy_fixtures: dict, recorder: _VersionRecorder) -> dict:
    import decision  # noqa: PLC0415

    fixture = load_fixture_for(case, legacy_fixtures)
    decision.get_repayment_history = lambda applicant_id: copy.deepcopy(fixture)  # BigQuery 대신 고정값
    recorder.install()
    recorder.versions.clear()
    r = decision.make_final_decision(
        case["applicant_id"],
        is_reevaluation=bool(case["is_reevaluation"]),
        business_description_override=case["business_description"],
    )
    return {
        "quant_tier": r.quant_tier,
        "default_probability": r.default_probability,
        "final_decision": r.final_decision,
        "adjustment_applied": r.adjustment_applied,
        "adjustment_direction": r.adjustment_direction,
        "decision_reasoning": r.decision_reasoning,
        "approved_amount_krw": r.approved_amount_krw,
        "business_summary": r.business_summary,
        "policy_citations": r.policy_citations,
        "tool_call_log": r.tool_call_log,
        "tool_call_summary": r.tool_call_summary,
        "critic_verdict": r.critic_verdict,
        "critic_reasoning": r.critic_reasoning,
        "critic_policy_violation": r.critic_policy_violation,
        "model_versions": sorted(set(recorder.versions)),
    }


def run_hardcap_case(hc: dict) -> dict:
    import bigquery_logger  # noqa: PLC0415
    import payment_mock  # noqa: PLC0415

    bigquery_logger.get_today_disbursed_total_krw = lambda: hc["today_disbursed_total_krw_fixture"]
    try:
        payment_mock._check_hard_caps(hc["synthetic_applicant_id"], hc["requested_loan_krw"])
        return {"blocked": False, "message": None}
    except payment_mock.FundControlError as e:
        msg = str(e)
        return {"blocked": True, "message": msg, "block_reason": "per_tx" if msg.startswith("건별") else "daily"}


def classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if "PerDay" in text:
        return "quota_daily"
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "rate_limit"
    if "503" in text or "UNAVAILABLE" in text or "500" in text or "INTERNAL" in text or "DEADLINE" in text:
        return "api_unavailable"
    if "record_decision을 호출하지 않고" in text:
        return "no_decision_recorded"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="1", help="실행할 회차 (예: 1 또는 2,3)")
    ap.add_argument("--max-cases", type=int, default=None, help="이번 실행에서 처리할 최대 케이스 수(디버그용)")
    args = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    runs = [int(x) for x in args.runs.split(",")]
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases, hc_cases = data["cases"], data["hard_cap_cases"]
    legacy_fixtures = json.loads(LEGACY_FIXTURE_PATH.read_text(encoding="utf-8"))
    manifest = check_manifest(allow_create=True)
    print(f"manifest OK (git_head_at_first_run={manifest['git_head_at_first_run'][:8]})")

    entries = load_entries()
    recorder = _VersionRecorder()
    processed = 0

    # 하드캡: 결정론적이므로 한 번만, 지연 없이
    for hc in hc_cases:
        done, _ = case_state(entries, "hardcap", 0, hc["case_id"])
        if done:
            continue
        try:
            res = run_hardcap_case(hc)
            entry = {"kind": "hardcap", "run": 0, "case_id": hc["case_id"], "attempt": 1, "status": "ok",
                     "started_at": datetime.now(timezone.utc).isoformat(), "result": res}
        except Exception as e:  # noqa: BLE001
            entry = {"kind": "hardcap", "run": 0, "case_id": hc["case_id"], "attempt": 1, "status": "error",
                     "started_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(e).__name__}: {e}",
                     "error_kind": "other"}
        append_entry(entry)
        entries.append(entry)
        print(f"[hardcap] {hc['case_id']}: {entry['status']} {entry.get('result', entry.get('error'))}")

    first_api_call = True
    for run in runs:
        for idx, case in enumerate(cases, start=1):
            done, attempts = case_state(entries, "agent", run, case["case_id"])
            if done:
                continue
            if attempts >= MAX_TOTAL_ATTEMPTS:
                print(f"[run{run} {idx}/{len(cases)}] {case['case_id']}: 누적 시도 {attempts}회 초과 — 오류로 확정")
                continue
            if args.max_cases is not None and processed >= args.max_cases:
                print("max-cases 도달 — 종료")
                return 0
            for k in range(MAX_ATTEMPTS_PER_INVOCATION):
                if attempts >= MAX_TOTAL_ATTEMPTS:
                    break
                if not first_api_call:
                    time.sleep(INTER_CASE_DELAY_SEC)
                first_api_call = False
                attempt_no = attempts + 1
                print(f"[run{run} {idx}/{len(cases)}] {case['case_id']} (신청자 {case['applicant_id']}, {case['category']}, "
                      f"시도 {attempt_no})", flush=True)
                t0 = time.time()
                started = datetime.now(timezone.utc).isoformat()
                try:
                    result = run_agent_case(case, legacy_fixtures, recorder)
                    entry = {"kind": "agent", "run": run, "case_id": case["case_id"], "attempt": attempt_no,
                             "status": "ok", "started_at": started, "elapsed_sec": round(time.time() - t0, 1),
                             "result": result}
                except Exception as e:  # noqa: BLE001 — 한 케이스 실패로 배치를 죽이지 않는다
                    entry = {"kind": "agent", "run": run, "case_id": case["case_id"], "attempt": attempt_no,
                             "status": "error", "started_at": started, "elapsed_sec": round(time.time() - t0, 1),
                             "error": f"{type(e).__name__}: {e}"[:2000], "error_kind": classify_error(e),
                             "traceback_tail": traceback.format_exc()[-1500:]}
                append_entry(entry)
                entries.append(entry)
                if entry["status"] == "ok":
                    r = entry["result"]
                    print(f"    -> {r['final_decision']} (정량 {r['quant_tier']}, Critic {r['critic_verdict']}, {entry['elapsed_sec']}s)")
                    processed += 1
                    break
                print(f"    !! 오류[{entry['error_kind']}]: {entry['error'][:200]}")
                if entry["error_kind"] == "quota_daily":
                    print("일일 할당량 소진 — 기록 후 종료합니다. 같은 명령으로 이어서 재개하세요.")
                    return EXIT_QUOTA
                attempts += 1
                processed += 1 if attempts >= MAX_TOTAL_ATTEMPTS else 0
    print("요청한 회차의 처리가 끝났습니다 (남은 미완료 케이스는 report에서 오류로 집계).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
