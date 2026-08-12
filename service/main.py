"""CreditFlow Agent — Cloud Run 서비스.

POST /underwrite/{applicant_id}
    Gemini 에이전트 판정(decision.py) -> devnet 집행(payment_mock.py) 실행 후,
    결과를 Pub/Sub(payment-events)에 발행한다.
    Eventarc가 이 토픽을 구독해 Workflow(영수증 발행 + BigQuery 로그)를 트리거한다.

GET /
    최근 심사 결과를 BigQuery에서 조회해 보여주는 간단한 상태 페이지 (라이브 데모 URL).
"""

import html
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "agent"))

from fastapi import FastAPI, Form, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402
from google.cloud import pubsub_v1  # noqa: E402

from decision import make_final_decision  # noqa: E402
import payment_mock  # noqa: E402
import live_state  # noqa: E402
from business_text import SAMPLE_BUSINESS_DESCRIPTIONS  # noqa: E402
from bigquery_logger import (  # noqa: E402
    get_client as get_bq_client,
    delete_decision,
    find_recent_execution,
    PROJECT_ID,
    DATASET_ID,
    TABLE_ID,
    RECEIPTS_TABLE_ID,
    REEVAL_TABLE_ID,
)

app = FastAPI(title="CreditFlow Agent")

PUBSUB_TOPIC = os.environ.get("PAYMENT_EVENTS_TOPIC", "payment-events")
# 같은 신청자를 최근 N분 내 중복 호출하면(재시도/실수 재클릭) devnet 송금이 두 번 나가는 걸 막는다.
# BigQuery 스트리밍 버퍼 지연으로 완벽하진 않음(best-effort) — 초 단위로 거의 동시에 들어오는
# 요청까지는 못 막지만, 실제로 겪은 문제(수동 재실행/재시도 스크립트)는 이 정도로 충분히 막힌다.
UNDERWRITE_IDEMPOTENCY_MINUTES = int(os.environ.get("UNDERWRITE_IDEMPOTENCY_MINUTES", "10"))
# 대시보드 삭제 버튼 + 데모용 심사 요청 버튼 보호용 — Cloud Run이 인증 없이 공개되어 있어서,
# 이 키를 아는 사람(발표자)만 실행할 수 있게 최소한의 장치를 둔다 (완전한 인증은 아님).
# 클릭 시점에 prompt()로 입력받아 서버로 보내며, HTML에는 값이 박히지 않는다.
DEMO_KEY = os.environ.get("DEMO_KEY", "")
_publisher = None


def get_publisher():
    global _publisher
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _publisher


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/underwrite/{applicant_id}")
def underwrite(applicant_id: int):
    try:
        existing = find_recent_execution(applicant_id, min_minutes=UNDERWRITE_IDEMPOTENCY_MINUTES)
    except Exception:  # noqa: BLE001
        # BigQuery 조회 자체가 실패하면 가드를 건너뛰고 정상 진행한다 (가용성 우선의 best-effort 가드).
        existing = None

    if existing:
        ts = existing.get("timestamp")
        return {
            "applicant_id": applicant_id,
            "decision": existing.get("decision"),
            "status": existing.get("status"),
            "tx_signature": existing.get("tx_signature"),
            "explorer_url": existing.get("explorer_url"),
            "approved_amount_krw": existing.get("requested_loan_krw"),
            "decision_reasoning": existing.get("rationale"),
            "idempotent_replay": True,
            "original_timestamp": ts.isoformat() if ts else None,
        }

    try:
        live_state.mark_in_progress(applicant_id)
    except Exception:  # noqa: BLE001
        pass  # 대시보드 "심사 중" 표시는 부가 기능 — 실패해도 심사 흐름은 계속 진행

    try:
        try:
            decision_result = make_final_decision(applicant_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"판정 실패: {e}")

        payment_result = payment_mock.disburse_loan(
            applicant_id=applicant_id,
            decision=decision_result.final_decision,
            requested_loan_krw=decision_result.approved_amount_krw,
            rationale=decision_result.decision_reasoning,
        )

        event = {
            "applicant_id": applicant_id,
            "decision": decision_result.final_decision,
            "status": payment_result.status,
            "tx_signature": payment_result.tx_signature,
            "explorer_url": payment_result.explorer_url,
            "approved_amount_krw": decision_result.approved_amount_krw,
        }

        try:
            publisher = get_publisher()
            topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
            publisher.publish(topic_path, json.dumps(event, ensure_ascii=False).encode("utf-8")).result(timeout=10)
        except Exception as e:  # noqa: BLE001
            # 이벤트 발행 실패해도 심사/집행 자체는 이미 완료된 상태이므로 응답은 정상 반환
            event["pubsub_error"] = str(e)

        return {
            **event,
            "quant_tier": decision_result.quant_tier,
            "default_probability": decision_result.default_probability,
            "adjustment_applied": decision_result.adjustment_applied,
            "decision_reasoning": decision_result.decision_reasoning,
            "feature_contributions": decision_result.feature_contributions,
        }
    finally:
        try:
            live_state.clear_in_progress(applicant_id)
        except Exception:  # noqa: BLE001
            pass


@app.post("/demo/underwrite")
def demo_underwrite(applicant_id: int = Form(...), key: str = Form(...)):
    """대시보드의 "심사 요청" 버튼 전용 진입점 — DEMO_KEY로 게이트된 /underwrite 래퍼.

    /underwrite/{applicant_id} 자체는 curl 데모/외부 연동 하위호환을 위해 그대로 공개로 둔다.
    """
    if not DEMO_KEY or key != DEMO_KEY:
        raise HTTPException(status_code=403, detail="진행 권한이 없습니다.")
    return underwrite(applicant_id)


@app.post("/jobs/reevaluate-due")
def reevaluate_due(min_days: int = 90):
    """Cloud Scheduler가 주기적으로 호출. min_days 이상 지난 조건부승인 건을 재심사하고,
    승인으로 상향된 건은 잔여 50%를 즉시 devnet에 집행한다."""
    from reevaluation import run_due_reevaluations

    try:
        results = run_due_reevaluations(min_days=min_days)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"재심사 실패: {e}")
    return {"processed": len(results), "results": results}


@app.post("/decisions/delete")
def delete_decision_endpoint(
    applicant_id: int = Form(...),
    timestamp: str = Form(...),
    key: str = Form(...),
):
    if not DEMO_KEY or key != DEMO_KEY:
        raise HTTPException(status_code=403, detail="삭제 권한이 없습니다.")

    result = delete_decision(applicant_id, timestamp)
    if not result["ok"]:
        from urllib.parse import quote

        return RedirectResponse(url=f"/?delete_error={quote(result['error'])}", status_code=303)
    return RedirectResponse(url="/", status_code=303)


_ORDER_COL_BY_TABLE = {
    TABLE_ID: "timestamp",
    RECEIPTS_TABLE_ID: "receipt_issued_at",
    REEVAL_TABLE_ID: "reevaluated_at",
}


def _fetch_recent(table_id: str, limit: int = 15) -> list[dict]:
    client = get_bq_client()
    order_col = _ORDER_COL_BY_TABLE.get(table_id, "timestamp")
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}` ORDER BY {order_col} DESC LIMIT {limit}"
    return [dict(row) for row in client.query(query).result()]


def _fetch_summary() -> dict:
    client = get_bq_client()
    query = f"""
        SELECT
          COUNT(*) AS total,
          COUNTIF(decision = 'approve') AS approved,
          COUNTIF(decision = 'conditional') AS conditional,
          COUNTIF(decision = 'reject') AS rejected,
          COUNTIF(status = 'EXECUTED') AS executed,
          SUM(IF(status = 'EXECUTED', devnet_test_amount, 0)) AS total_disbursed
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    rows = list(client.query(query).result())
    return dict(rows[0]) if rows else {}


_DECISION_KR = {"approve": "승인", "conditional": "조건부승인", "reject": "거절"}
_DECISION_BADGE_CLASS = {"approve": "badge-approve", "conditional": "badge-conditional", "reject": "badge-reject"}


def _badge(decision: str) -> str:
    label = _DECISION_KR.get(decision, decision or "-")
    cls = _DECISION_BADGE_CLASS.get(decision, "badge-neutral")
    return f'<span class="badge {cls}">{label}</span>'


def _tx_link(tx_signature, explorer_url) -> str:
    if not tx_signature:
        return '<span class="muted">-</span>'
    short = f"{tx_signature[:8]}…{tx_signature[-6:]}"
    url = explorer_url or f"https://explorer.solana.com/tx/{tx_signature}?cluster=devnet"
    return f'<a class="txlink" href="{url}" target="_blank" rel="noopener">{short}</a>'


def _delete_form_html(applicant_id, ts) -> str:
    if not DEMO_KEY or not ts:
        return '<span class="muted">-</span>'
    ts_iso = ts.isoformat()
    # DEMO_KEY를 HTML에 값으로 박아두지 않는다 — view-source로 그대로 유출되는 걸 막기 위해
    # 클릭 시점에 prompt()로 입력받아 hidden input에 채운 뒤에만 제출한다.
    onsubmit = (
        f"if(!confirm('신청자 {applicant_id}번 기록을 삭제할까요? 되돌릴 수 없습니다.')) return false; "
        "var k = getDemoKey(); "
        "if (!k) return false; "
        "this.key.value = k; "
        "return true;"
    )
    return f"""<form method="post" action="/decisions/delete" onsubmit="{onsubmit}" style="display:inline">
  <input type="hidden" name="applicant_id" value="{applicant_id}">
  <input type="hidden" name="timestamp" value="{ts_iso}">
  <input type="hidden" name="key" value="">
  <button type="submit" class="del-btn">삭제</button>
</form>"""


def _decisions_table_html(records: list[dict]) -> str:
    if not records:
        return '<tr><td colspan="7" class="empty">아직 심사 기록이 없습니다.</td></tr>'
    rows = []
    for r in records:
        amount = r.get("devnet_test_amount")
        currency = r.get("currency") or ""
        amount_str = f"{amount:.2f} {currency}" if amount else '<span class="muted">-</span>'
        ts = r.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M UTC") if ts else "-"
        rows.append(
            "<tr>"
            f'<td class="mono">{r.get("applicant_id", "-")}</td>'
            f"<td>{_badge(r.get('decision'))}</td>"
            f'<td class="mono">{r.get("requested_loan_krw", 0):,}원</td>'
            f'<td class="mono">{amount_str}</td>'
            f"<td>{_tx_link(r.get('tx_signature'), r.get('explorer_url'))}</td>"
            f'<td class="muted">{ts_str}</td>'
            f"<td>{_delete_form_html(r.get('applicant_id'), ts)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _reeval_table_html(records: list[dict]) -> str:
    if not records:
        return '<tr><td colspan="6" class="empty">아직 재심사 기록이 없습니다.</td></tr>'
    rows = []
    for r in records:
        upgraded = r.get("upgraded")
        result_badge = (
            '<span class="badge badge-approve">승인 상향</span>'
            if upgraded
            else '<span class="badge badge-neutral">유지</span>'
        )
        amount = r.get("additional_amount")
        currency = r.get("currency") or ""
        amount_str = f"{amount:.2f} {currency}" if amount else '<span class="muted">-</span>'
        ts = r.get("reevaluated_at")
        ts_str = ts.strftime("%Y-%m-%d %H:%M UTC") if ts else "-"
        rows.append(
            "<tr>"
            f'<td class="mono">{r.get("applicant_id", "-")}</td>'
            f'<td>{_badge(r.get("original_decision"))} → {_badge(r.get("new_decision"))}</td>'
            f"<td>{result_badge}</td>"
            f'<td class="mono">{amount_str}</td>'
            f"<td>{_tx_link(r.get('tx_signature'), r.get('explorer_url'))}</td>"
            f'<td class="muted">{ts_str}</td>'
            "</tr>"
        )
    return "".join(rows)


def _in_progress_html(in_progress: list[dict]) -> str:
    if not in_progress:
        return ""
    items = "".join(
        f'<div class="progress-item">⏳ 신청자 <span class="mono">{p.get("applicant_id")}</span>번 심사 중…</div>'
        for p in in_progress
    )
    return f'<div class="progress-banner">{items}</div>'


def _applicant_options_html() -> str:
    options = []
    for aid, text in SAMPLE_BUSINESS_DESCRIPTIONS.items():
        label = html.escape(text[:24].strip())
        options.append(f'<option value="{aid}">{aid} — {label}…</option>')
    return "".join(options)


def _render_live_region(
    decisions: list[dict],
    summary: dict,
    reevaluations: list[dict],
    in_progress: list[dict],
    fetch_error: str = None,
    delete_error: str = None,
) -> str:
    """폴링으로 계속 갱신되는 대시보드 영역(KPI/최근 심사/재심사)을 렌더링한다.

    GET / (최초 로드)와 GET /live-region (2.5초 폴링) 양쪽에서 동일하게 사용한다.
    """
    total = summary.get("total") or 0
    approved = summary.get("approved") or 0
    conditional = summary.get("conditional") or 0
    rejected = summary.get("rejected") or 0
    executed = summary.get("executed") or 0
    total_disbursed = summary.get("total_disbursed") or 0
    approval_rate = f"{(approved / total * 100):.0f}%" if total else "-"

    error_banner = (
        f'<div class="banner">⚠ BigQuery 조회 실패: {fetch_error}</div>' if fetch_error else ""
    )
    if delete_error:
        error_banner += f'<div class="banner">⚠ 삭제 실패: {delete_error}</div>'

    return f"""
    {error_banner}
    {_in_progress_html(in_progress)}

    <section class="kpi-row">
      <div class="kpi">
        <div class="kpi-label">총 심사 건수</div>
        <div class="kpi-value">{total}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">승인율</div>
        <div class="kpi-value accent">{approval_rate}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">온체인 집행 건수</div>
        <div class="kpi-value">{executed}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">누적 집행액 (devnet)</div>
        <div class="kpi-value">{total_disbursed:.2f} USDC</div>
      </div>
    </section>

    <section class="card">
      <div class="card-head">
        <div class="card-title">최근 심사 결과</div>
        <div class="card-note">승인 {approved} · 조건부 {conditional} · 거절 {rejected}</div>
      </div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>신청자 ID</th><th>판정</th><th>대출한도(KRW)</th><th>devnet 집행액</th><th>tx</th><th>시각</th><th></th></tr></thead>
          <tbody>{_decisions_table_html(decisions)}</tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <div class="card-head">
        <div class="card-title">재심사 결과</div>
        <div class="card-note">조건부승인 건의 자동 재심사(Cloud Scheduler) 처리 이력</div>
      </div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>신청자 ID</th><th>판정 변화</th><th>결과</th><th>추가 집행액</th><th>tx</th><th>재심사 시각</th></tr></thead>
          <tbody>{_reeval_table_html(reevaluations)}</tbody>
        </table>
      </div>
    </section>
    """


def _load_dashboard_data(delete_error: str = None) -> str:
    try:
        decisions = _fetch_recent(TABLE_ID)
        summary = _fetch_summary()
        fetch_error = None
    except Exception as e:  # noqa: BLE001
        decisions, summary, fetch_error = [], {}, str(e)

    try:
        reevaluations = _fetch_recent(REEVAL_TABLE_ID)
    except Exception:  # noqa: BLE001
        reevaluations = []

    try:
        in_progress = live_state.list_in_progress()
    except Exception:  # noqa: BLE001
        in_progress = []

    return _render_live_region(decisions, summary, reevaluations, in_progress, fetch_error, delete_error)


@app.get("/live-region", response_class=HTMLResponse)
def live_region():
    """대시보드가 2.5초마다 폴링하는 조각 HTML — 다른 접속자가 켜둔 화면에도 "심사 중" 상태가 뜨게 한다."""
    return _load_dashboard_data()


@app.get("/", response_class=HTMLResponse)
def status_page(delete_error: str = None):
    live_region_html = _load_dashboard_data(delete_error)

    return f"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CreditFlow Agent — 심사 대시보드</title>
<style>
  :root {{
    color-scheme: light;
    --page: #f4f6fa; --surface: #ffffff; --surface-2: #f8f9fc;
    --ink: #14171c; --ink-2: #52586b; --muted: #8a8d99;
    --border: rgba(20,23,28,0.08);
    --accent: #2a78d6; --accent-soft: #e8f1fc;
    --good: #0ca30c; --good-soft: #e7f7e7;
    --warn: #b8860b; --warn-soft: #fdf3d9;
    --bad: #d03b3b; --bad-soft: #fbe9e9;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --page: #0d0e12; --surface: #17191f; --surface-2: #1c1f26;
      --ink: #f2f3f5; --ink-2: #b7bcc9; --muted: #82869a;
      --border: rgba(255,255,255,0.08);
      --accent: #4c93e8; --accent-soft: #16283f;
      --good: #34c759; --good-soft: #102417;
      --warn: #e0b23a; --warn-soft: #2c2410;
      --bad: #e5605f; --bad-soft: #2c1616;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 24px 64px; background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 1040px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }}
  .eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--accent); }}
  h1 {{ font-size: 22px; font-weight: 800; margin: 6px 0 4px; }}
  .subtitle {{ color: var(--ink-2); font-size: 13.5px; }}
  .banner {{ background: var(--bad-soft); color: var(--bad); padding: 10px 14px; border-radius: 8px; font-size: 13px; }}

  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .kpi-label {{ font-size: 12px; color: var(--ink-2); }}
  .kpi-value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .kpi-value.accent {{ color: var(--accent); }}

  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
  .card-title {{ font-size: 14.5px; font-weight: 700; }}
  .card-note {{ font-size: 12px; color: var(--muted); }}

  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  th {{ text-align: left; font-weight: 600; color: var(--ink-2); padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 11.5px; text-transform: uppercase; letter-spacing: .03em; }}
  td {{ padding: 9px 10px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  .mono {{ font-variant-numeric: tabular-nums; }}
  .muted {{ color: var(--muted); }}
  .empty {{ text-align: center; color: var(--muted); padding: 20px; }}

  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }}
  .badge-approve {{ background: var(--good-soft); color: var(--good); }}
  .badge-conditional {{ background: var(--warn-soft); color: var(--warn); }}
  .badge-reject {{ background: var(--bad-soft); color: var(--bad); }}
  .badge-neutral {{ background: var(--surface-2); color: var(--muted); }}

  .txlink {{ color: var(--accent); text-decoration: none; font-variant-numeric: tabular-nums; }}
  .txlink:hover {{ text-decoration: underline; }}

  .del-btn {{
    background: var(--bad-soft); color: var(--bad); border: 1px solid transparent;
    border-radius: 6px; padding: 4px 10px; font-size: 11.5px; font-weight: 600; cursor: pointer;
  }}
  .del-btn:hover {{ border-color: var(--bad); }}

  .action-hint {{ font-size: 12.5px; color: var(--ink-2); }}
  code {{ background: var(--surface-2); padding: 1px 6px; border-radius: 4px; font-size: 12px; }}

  .progress-banner {{
    background: var(--accent-soft); border-radius: 8px; padding: 10px 14px;
    display: flex; flex-direction: column; gap: 4px;
  }}
  .progress-item {{ color: var(--accent); font-size: 13px; font-weight: 600; }}

  .demo-form {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .demo-form select {{
    background: var(--surface-2); color: var(--ink); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 10px; font-size: 13px; max-width: 320px;
  }}
  .run-btn {{
    background: var(--accent); color: #fff; border: none; border-radius: 6px;
    padding: 7px 16px; font-size: 13px; font-weight: 600; cursor: pointer;
  }}
  .run-btn:hover {{ opacity: 0.9; }}
  .demo-status {{ font-size: 12.5px; color: var(--ink-2); }}

  @media (max-width: 720px) {{ .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="eyebrow">CreditFlow Agent · Live PoC</div>
      <h1>소상공인 대출 사전심사 심사 대시보드</h1>
      <div class="subtitle">Gemini 판정 + Solana devnet 자동 집행 결과를 실시간으로 조회합니다 (읽기 전용)</div>
    </header>

    <div id="live-region">{live_region_html}</div>

    <section class="card">
      <div class="card-head">
        <div class="card-title">새 심사 실행</div>
        <div class="card-note">진행 키 필요 (발표자 전용)</div>
      </div>
      <form id="demo-underwrite-form" class="demo-form">
        <select name="applicant_id" id="demo-applicant-select">
          {_applicant_options_html()}
        </select>
        <button type="submit" class="run-btn">심사 요청</button>
        <span id="demo-status" class="demo-status"></span>
      </form>
      <div class="action-hint" style="margin-top:10px">
        또는 API로 직접 호출: <code>POST /underwrite/{{applicant_id}}</code> (판정 후 승인 건은 devnet USDC 집행까지 자동 수행)<br>
        예: <code>curl.exe -s -X POST -d "{{}}" https://creditflow-agent-46585987317.asia-northeast3.run.app/underwrite/10736</code>
      </div>
    </section>
  </div>

  <script>
    function getDemoKey() {{
      let k = sessionStorage.getItem('demoKey');
      if (!k) {{
        k = prompt('데모 키를 입력하세요');
        if (k) sessionStorage.setItem('demoKey', k);
      }}
      return k;
    }}

    async function refreshLiveRegion() {{
      try {{
        const res = await fetch('/live-region');
        if (res.ok) {{
          document.getElementById('live-region').innerHTML = await res.text();
        }}
      }} catch (e) {{ /* 폴링 실패는 조용히 무시하고 다음 주기에 재시도 */ }}
    }}

    document.getElementById('demo-underwrite-form').addEventListener('submit', async function (e) {{
      e.preventDefault();
      const key = getDemoKey();
      if (!key) return;
      const applicantId = document.getElementById('demo-applicant-select').value;
      const statusEl = document.getElementById('demo-status');
      statusEl.textContent = '요청 접수 — 심사 중…';
      const body = new URLSearchParams({{applicant_id: applicantId, key: key}});
      try {{
        const res = await fetch('/demo/underwrite', {{method: 'POST', body: body}});
        if (res.status === 403) {{
          sessionStorage.removeItem('demoKey');
          statusEl.textContent = '키가 틀렸습니다. 다시 시도해 주세요.';
          return;
        }}
        if (!res.ok) {{
          statusEl.textContent = '심사 요청 실패';
          return;
        }}
        statusEl.textContent = '완료됨';
        refreshLiveRegion();
      }} catch (err) {{
        statusEl.textContent = '요청 중 오류가 발생했습니다.';
      }}
    }});

    refreshLiveRegion();
    setInterval(refreshLiveRegion, 2500);
  </script>
</body>
</html>
"""
