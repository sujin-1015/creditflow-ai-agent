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
import critic  # noqa: E402
import payment_mock  # noqa: E402
import live_state  # noqa: E402
from business_text import SAMPLE_BUSINESS_DESCRIPTIONS, SAMPLE_BUSINESS_INDUSTRY  # noqa: E402
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

        try:
            payment_result = payment_mock.disburse_loan(
                applicant_id=applicant_id,
                decision=decision_result.final_decision,
                requested_loan_krw=decision_result.approved_amount_krw,
                rationale=decision_result.decision_reasoning,
                critic_verdict=decision_result.critic_verdict,
                critic_reasoning=decision_result.critic_reasoning,
                tool_call_summary=decision_result.tool_call_summary,
            )
        except payment_mock.FundControlError as e:
            raise HTTPException(
                status_code=403,
                detail=f"판정은 '{decision_result.final_decision}'이었으나 자금 통제(하드 캡)에 의해 집행이 차단됐습니다: {e}",
            )
        except payment_mock.DisbursementError as e:
            wallet_note = ""
            if e.wallet_address:
                tag = "새로 발급된" if e.wallet_newly_issued else "기존"
                wallet_note = f" (참고: {tag} 임베디드 지갑은 정상 발급/확인됨 — {e.wallet_address})"
            raise HTTPException(
                status_code=500,
                detail=f"판정은 '{decision_result.final_decision}'으로 나왔으나 devnet 집행에 실패했습니다: {e}{wallet_note}",
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"판정은 '{decision_result.final_decision}'으로 나왔으나 devnet 집행에 실패했습니다: {e}",
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
            "tool_call_log": decision_result.tool_call_log,
            "tool_call_summary": decision_result.tool_call_summary,
            "critic_verdict": decision_result.critic_verdict,
            "critic_reasoning": decision_result.critic_reasoning,
            "critic_policy_violation": decision_result.critic_policy_violation,
            "wallet_address": payment_result.wallet_address,
            "wallet_newly_issued": payment_result.wallet_newly_issued,
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


@app.post("/demo/repay")
def demo_repay(applicant_id: int = Form(...), key: str = Form(...)):
    """상환 실행 — 지급의 역방향(신청자 지갑 -> treasury). 실제 devnet 트랜잭션이며, loan_decisions와는 별도인
    repayments 테이블에만 기록되고, 대시보드의 심사 이력/재심사 이력에는 영향을 주지 않는다."""
    if not DEMO_KEY or key != DEMO_KEY:
        raise HTTPException(status_code=403, detail="진행 권한이 없습니다.")
    try:
        result = payment_mock.collect_repayment(applicant_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"상환 처리 실패: {e}")
    return {
        "applicant_id": result.applicant_id,
        "amount_usdc": result.amount_usdc,
        "tx_signature": result.tx_signature,
        "explorer_url": result.explorer_url,
        "status": result.status,
    }


@app.post("/demo/hard-cap-test")
def demo_hard_cap_test(scenario: str = Form(...), key: str = Form(...)):
    """하드 캡 단독 데모 — 지갑 발급이나 devnet 송금 없이 payment_mock._check_hard_caps()만
    실행해, 건별 한도 초과 요청이 실제로 그 자리에서 차단되는지 즉시 보여준다."""
    if not DEMO_KEY or key != DEMO_KEY:
        raise HTTPException(status_code=403, detail="진행 권한이 없습니다.")
    if scenario not in payment_mock.HARD_CAP_DEMO_SCENARIOS:
        raise HTTPException(status_code=400, detail="알 수 없는 시나리오입니다.")

    amount = payment_mock.HARD_CAP_DEMO_SCENARIOS[scenario]
    try:
        payment_mock._check_hard_caps(payment_mock.HARD_CAP_DEMO_APPLICANT_ID, amount)
        return {
            "scenario": scenario,
            "requested_krw": amount,
            "blocked": False,
            "message": f"{amount:,}원 요청 — 하드 캡(건별 {payment_mock.PER_TX_HARD_CAP_KRW:,}원) 이내라 정상 통과했습니다.",
        }
    except payment_mock.FundControlError as e:
        return {"scenario": scenario, "requested_krw": amount, "blocked": True, "message": str(e)}


@app.post("/demo/critic-test")
def demo_critic_test(scenario: str = Form(...), key: str = Form(...)):
    """Critic Agent 단독 데모 — 1차 판정 에이전트를 거치지 않고, 미리 준비된 시나리오(정책
    위반/정상)를 바로 Critic에게 넘겨 실제로 반박/승인하는지 확인한다. 판정 데이터를 새로
    만들거나 저장하지 않는 순수 조회성 테스트라 BigQuery/devnet에는 아무 영향이 없다."""
    if not DEMO_KEY or key != DEMO_KEY:
        raise HTTPException(status_code=403, detail="진행 권한이 없습니다.")
    if scenario not in critic.DEMO_SCENARIOS:
        raise HTTPException(status_code=400, detail="알 수 없는 시나리오입니다.")
    try:
        result = critic.review_decision(**critic.DEMO_SCENARIOS[scenario])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Critic 테스트 실패: {e}")
    return {"scenario": scenario, **result}


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

# 순수 디자인용 인라인 아이콘(Feather Icons 스타일, MIT). 로직에는 영향 없음.
_ICON_TOTAL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'
_ICON_APPROVAL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
_ICON_CHAIN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>'
_ICON_DISBURSED = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>'
_ICON_BOLT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'


def _badge(decision: str) -> str:
    label = _DECISION_KR.get(decision, decision or "-")
    cls = _DECISION_BADGE_CLASS.get(decision, "badge-neutral")
    return f'<span class="badge {cls}"><span class="badge-dot"></span>{label}</span>'


_CRITIC_KR = {"approve": "승인", "reject": "반박"}
_CRITIC_BADGE_CLASS = {"approve": "badge-approve", "reject": "badge-reject"}


def _critic_badge(verdict: str, reasoning: str = None) -> str:
    if not verdict:
        return '<span class="muted">-</span>'
    label = _CRITIC_KR.get(verdict, verdict)
    cls = _CRITIC_BADGE_CLASS.get(verdict, "badge-neutral")
    title = html.escape(reasoning or "", quote=True)
    return f'<span class="badge {cls}" title="{title}"><span class="badge-dot"></span>{label}</span>'


def _tool_log_html(summary: str) -> str:
    """에이전트가 자율적으로 고른 도구 호출 순서 — 신청자마다 다르게 나온다는 것 자체가
    "코드가 아니라 모델이 순서를 정한다"는 증거라 대시보드에 그대로 노출한다."""
    if not summary:
        return '<span class="muted">-</span>'
    steps = [s.strip() for s in summary.split("→")]
    escaped = html.escape(" → ".join(steps), quote=True)
    return f'<span class="tool-log" title="{escaped}">{escaped}</span>'


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
        return '<tr><td colspan="9" class="empty">아직 심사 기록이 없습니다.</td></tr>'
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
            f"<td>{_critic_badge(r.get('critic_verdict'), r.get('critic_reasoning'))}</td>"
            f"<td>{_tool_log_html(r.get('tool_call_summary'))}</td>"
            f'<td class="muted">{ts_str}</td>'
            f"<td>{_delete_form_html(r.get('applicant_id'), ts)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _reeval_table_html(records: list[dict]) -> str:
    if not records:
        return '<tr><td colspan="8" class="empty">아직 재심사 기록이 없습니다.</td></tr>'
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
            f"<td>{_critic_badge(r.get('critic_verdict'), r.get('critic_reasoning'))}</td>"
            f"<td>{_tool_log_html(r.get('tool_call_summary'))}</td>"
            f'<td class="muted">{ts_str}</td>'
            "</tr>"
        )
    return "".join(rows)


def _in_progress_html(in_progress: list[dict]) -> str:
    if not in_progress:
        return ""
    items = "".join(
        f'<div class="progress-item"><span class="pulse-dot"></span>신청자 '
        f'<span class="mono">{p.get("applicant_id")}</span>번 심사 중…</div>'
        for p in in_progress
    )
    return f'<div class="progress-banner">{items}</div>'


def _applicant_options_html() -> str:
    options = []
    for aid in SAMPLE_BUSINESS_DESCRIPTIONS:
        label = html.escape(SAMPLE_BUSINESS_INDUSTRY.get(aid, "업종 정보 없음"))
        options.append(f'<option value="{aid}">{aid} — {label}</option>')
    return "".join(options)


def _render_live_region(
    decisions: list[dict],
    summary: dict,
    reevaluations: list[dict],
    in_progress: list[dict],
    fetch_error: str = None,
    delete_error: str = None,
) -> tuple[str, str]:
    """폴링으로 계속 갱신되는 대시보드 영역(KPI/최근 심사/재심사)을 렌더링한다.

    (top, bottom) 튜플로 나눠서 반환한다 — 그 사이에 "새 심사 실행" 폼(폴링 대상이 아닌 정적
    요소)을 끼워 넣기 위함. 폼까지 통째로 폴링 영역에 넣으면 2.5초마다 innerHTML이 갈아치워지며
    드롭다운 선택값이 초기화되고 이벤트 바인딩도 끊길 수 있어, 폼은 항상 두 영역 밖에 둔다.

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

    top = f"""
    {error_banner}
    {_in_progress_html(in_progress)}

    <section class="kpi-row">
      <div class="kpi">
        <div class="kpi-icon kpi-icon-accent">{_ICON_TOTAL}</div>
        <div class="kpi-body">
          <div class="kpi-label">총 심사 건수</div>
          <div class="kpi-value">{total}</div>
        </div>
      </div>
      <div class="kpi">
        <div class="kpi-icon kpi-icon-good">{_ICON_APPROVAL}</div>
        <div class="kpi-body">
          <div class="kpi-label">승인율</div>
          <div class="kpi-value accent">{approval_rate}</div>
        </div>
      </div>
      <div class="kpi">
        <div class="kpi-icon kpi-icon-violet">{_ICON_CHAIN}</div>
        <div class="kpi-body">
          <div class="kpi-label">온체인 집행 건수</div>
          <div class="kpi-value">{executed}</div>
        </div>
      </div>
      <div class="kpi">
        <div class="kpi-icon kpi-icon-warn">{_ICON_DISBURSED}</div>
        <div class="kpi-body">
          <div class="kpi-label">누적 집행액 (devnet)</div>
          <div class="kpi-value">{total_disbursed:.2f} USDC</div>
        </div>
      </div>
    </section>
    """

    bottom = f"""
    <section class="card">
      <div class="card-head">
        <div class="card-title">최근 심사 결과</div>
        <div class="card-note">승인 {approved} · 조건부 {conditional} · 거절 {rejected}</div>
      </div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>신청자 ID</th><th>판정</th><th>대출한도(KRW)</th><th>devnet 집행액</th><th>tx</th><th>Critic 검증</th><th>에이전트 도구 호출 순서</th><th>시각</th><th></th></tr></thead>
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
          <thead><tr><th>신청자 ID</th><th>판정 변화</th><th>결과</th><th>추가 집행액</th><th>tx</th><th>Critic 검증</th><th>에이전트 도구 호출 순서</th><th>재심사 시각</th></tr></thead>
          <tbody>{_reeval_table_html(reevaluations)}</tbody>
        </table>
      </div>
    </section>
    """

    return top, bottom


def _load_dashboard_data(delete_error: str = None) -> tuple[str, str]:
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


_LIVE_REGION_SPLIT = "<!--LIVE-REGION-SPLIT-->"


@app.get("/live-region", response_class=HTMLResponse)
def live_region():
    """대시보드가 2.5초마다 폴링하는 조각 HTML — 다른 접속자가 켜둔 화면에도 "심사 중" 상태가 뜨게 한다.

    top/bottom 두 조각을 구분자로 이어붙여 반환한다 (프론트에서 split해서 각자 다른
    컨테이너에 꽂는다 — "새 심사 실행" 폼이 그 사이에 고정으로 위치하기 때문)."""
    top, bottom = _load_dashboard_data()
    return f"{top}{_LIVE_REGION_SPLIT}{bottom}"


@app.get("/", response_class=HTMLResponse)
def status_page(delete_error: str = None):
    live_top, live_bottom = _load_dashboard_data(delete_error)

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
    --accent: #2a78d6; --accent-2: #7c5cff; --accent-soft: #e8f1fc;
    --good: #0ca30c; --good-soft: #e7f7e7;
    --warn: #b8860b; --warn-soft: #fdf3d9;
    --bad: #d03b3b; --bad-soft: #fbe9e9;
    --violet: #7c5cff; --violet-soft: #efeaff;
    --shadow-sm: 0 1px 2px rgba(20,23,28,0.05);
    --shadow-md: 0 10px 30px -12px rgba(20,23,28,0.18);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --page: #0a0b0f; --surface: #16181f; --surface-2: #1b1e26;
      --ink: #f2f3f5; --ink-2: #b7bcc9; --muted: #82869a;
      --border: rgba(255,255,255,0.09);
      --accent: #5b9cf0; --accent-2: #9a83ff; --accent-soft: #172440;
      --good: #34c759; --good-soft: #10241a;
      --warn: #e0b23a; --warn-soft: #2c2410;
      --bad: #e5605f; --bad-soft: #2c1616;
      --violet: #9a83ff; --violet-soft: #1f1a38;
      --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
      --shadow-md: 0 14px 32px -14px rgba(0,0,0,0.6);
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 40px 24px 72px; color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background:
      radial-gradient(720px 320px at 12% -8%, var(--accent-soft), transparent 60%),
      radial-gradient(640px 280px at 100% 0%, var(--violet-soft), transparent 55%),
      var(--page);
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 26px; }}
  #live-region-top, #live-region-bottom {{ display: flex; flex-direction: column; gap: 26px; }}

  .page-header {{ display: flex; align-items: flex-start; gap: 16px; }}
  .logo-mark {{
    flex: none; width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; color: #fff;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: var(--shadow-md);
  }}
  .logo-mark svg {{ width: 22px; height: 22px; }}
  .header-text {{ min-width: 0; }}

  .eyebrow {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11.5px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
    color: var(--accent); background: var(--accent-soft); border-radius: 999px;
    padding: 4px 10px 4px 8px;
  }}
  .eyebrow-dot {{
    width: 6px; height: 6px; border-radius: 50%; background: var(--good);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--good) 25%, transparent);
  }}
  h1 {{ font-size: 23px; font-weight: 800; margin: 10px 0 4px; letter-spacing: -.01em; }}
  .subtitle {{ color: var(--ink-2); font-size: 13.5px; }}
  .banner {{ background: var(--bad-soft); color: var(--bad); padding: 10px 14px; border-radius: 8px; font-size: 13px; }}

  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
  .kpi {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px; box-shadow: var(--shadow-sm);
    display: flex; align-items: center; gap: 12px;
    transition: transform .15s ease, box-shadow .15s ease;
  }}
  .kpi:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-md); }}
  .kpi-icon {{
    flex: none; width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
  }}
  .kpi-icon svg {{ width: 18px; height: 18px; }}
  .kpi-icon-accent {{ background: var(--accent-soft); color: var(--accent); }}
  .kpi-icon-good {{ background: var(--good-soft); color: var(--good); }}
  .kpi-icon-violet {{ background: var(--violet-soft); color: var(--violet); }}
  .kpi-icon-warn {{ background: var(--warn-soft); color: var(--warn); }}
  .kpi-body {{ min-width: 0; }}
  .kpi-label {{ font-size: 12px; color: var(--ink-2); }}
  .kpi-value {{ font-size: 21px; font-weight: 800; margin-top: 2px; font-variant-numeric: tabular-nums; }}
  .kpi-value.accent {{ color: var(--accent); }}

  .card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 20px 22px; box-shadow: var(--shadow-sm);
  }}
  .card-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px; }}
  .card-title {{ font-size: 14.5px; font-weight: 700; }}
  .card-note {{ font-size: 12px; color: var(--muted); }}

  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  th {{ text-align: left; font-weight: 600; color: var(--ink-2); padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 11.5px; text-transform: uppercase; letter-spacing: .03em; }}
  td {{ padding: 10px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tbody tr {{ transition: background .1s ease; }}
  tbody tr:nth-child(even) {{ background: var(--surface-2); }}
  tbody tr:hover {{ background: var(--accent-soft); }}
  .mono {{ font-variant-numeric: tabular-nums; }}
  .muted {{ color: var(--muted); }}
  .empty {{ text-align: center; color: var(--muted); padding: 20px; }}

  .badge {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 600;
  }}
  .badge-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: .8; }}
  .badge-approve {{ background: var(--good-soft); color: var(--good); }}
  .badge-conditional {{ background: var(--warn-soft); color: var(--warn); }}
  .badge-reject {{ background: var(--bad-soft); color: var(--bad); }}
  .badge-neutral {{ background: var(--surface-2); color: var(--muted); }}

  .txlink {{ color: var(--accent); text-decoration: none; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .txlink:hover {{ text-decoration: underline; }}

  .tool-log {{
    display: inline-block; max-width: 260px; white-space: normal; word-break: break-word;
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
    font-size: 11px; color: var(--ink-2); line-height: 1.5;
  }}

  .del-btn {{
    background: var(--bad-soft); color: var(--bad); border: 1px solid transparent;
    border-radius: 8px; padding: 5px 12px; font-size: 11.5px; font-weight: 600; cursor: pointer;
    transition: background .15s ease, border-color .15s ease;
  }}
  .del-btn:hover {{ border-color: var(--bad); background: var(--bad); color: #fff; }}

  .action-hint {{ font-size: 12.5px; color: var(--ink-2); line-height: 1.7; }}
  code {{ background: var(--surface-2); padding: 1px 6px; border-radius: 4px; font-size: 12px; }}

  .progress-banner {{
    background: linear-gradient(135deg, var(--accent-soft), var(--violet-soft));
    border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px;
    display: flex; flex-direction: column; gap: 6px;
  }}
  .progress-item {{ display: flex; align-items: center; gap: 8px; color: var(--accent); font-size: 13px; font-weight: 700; }}
  .pulse-dot {{
    flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
    animation: pulse 1.4s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 45%, transparent); }}
    70% {{ box-shadow: 0 0 0 8px color-mix(in srgb, var(--accent) 0%, transparent); }}
    100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 0%, transparent); }}
  }}

  .demo-form {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .demo-form select {{
    background: var(--surface-2); color: var(--ink); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 12px; font-size: 13px; width: 460px; max-width: 100%;
    transition: border-color .15s ease, box-shadow .15s ease;
  }}
  .demo-form select:focus {{
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
  }}
  .run-btn {{
    background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff; border: none;
    border-radius: 8px; padding: 8px 18px; font-size: 13px; font-weight: 700; cursor: pointer;
    box-shadow: var(--shadow-sm);
    transition: transform .12s ease, box-shadow .12s ease;
  }}
  .run-btn:hover {{ transform: translateY(-1px); box-shadow: var(--shadow-md); }}
  .run-btn:active {{ transform: translateY(0); }}
  .demo-status {{ font-size: 12.5px; color: var(--ink-2); }}

  .critic-test-result {{
    display: none; margin-top: 14px; padding: 14px 16px; border-radius: 10px;
    background: var(--surface-2); border: 1px solid var(--border); font-size: 13px; line-height: 1.6;
  }}

  @media (max-width: 720px) {{ .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
  <div class="wrap">
    <header class="page-header">
      <div class="logo-mark">{_ICON_BOLT}</div>
      <div class="header-text">
        <div class="eyebrow"><span class="eyebrow-dot"></span>CreditFlow Agent · Live PoC</div>
        <h1>소상공인 대출 심사 에이전트 대시보드</h1>
        <div class="subtitle">Gemini 판정 + Solana devnet 자동 집행을 실시간으로 실행하고 결과를 조회합니다</div>
      </div>
    </header>

    <div id="live-region-top">{live_top}</div>

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
        승인/조건부승인 시 지갑이 없는 신청자에게는 임베디드(Passkey 방식) 지갑을 자동 발급한 뒤 즉시 집행합니다 — 시드구문 불필요, 수수료는 전부 서비스가 부담(Gasless)합니다.<br>
        또는 API로 직접 호출: <code>POST /underwrite/{{applicant_id}}</code> (판정 후 승인 건은 devnet USDC 집행까지 자동 수행)<br>
        예: <code>curl.exe -s -X POST -d "{{}}" https://creditflow-agent-46585987317.asia-northeast3.run.app/underwrite/10736</code>
      </div>
    </section>

    <div id="live-region-bottom">{live_bottom}</div>

    <section class="card">
      <div class="card-head">
        <div class="card-title">Critic Agent 단독 테스트</div>
        <div class="card-note">1차 판정 없이 Critic만 실행 — 정책 위반을 실제로 잡아내는지 확인 (진행 키 필요)</div>
      </div>
      <form id="critic-test-form" class="demo-form">
        <select name="scenario" id="critic-scenario-select">
          <option value="violation">위반 시나리오 — 정량 reject인데 정성 조정으로 승인 상향</option>
          <option value="clean">정상 시나리오 — 정책과 부합하는 conditional 유지</option>
        </select>
        <button type="submit" class="run-btn">테스트 실행</button>
      </form>
      <div id="critic-test-result" class="critic-test-result"></div>
    </section>

    <section class="card">
      <div class="card-head">
        <div class="card-title">상환 실행</div>
        <div class="card-note">지급의 역방향 — 신청자 지갑에서 treasury로 상환 (진행 키 필요)</div>
      </div>
      <form id="repay-form" class="demo-form">
        <select name="applicant_id" id="repay-applicant-select">
          {_applicant_options_html()}
        </select>
        <button type="submit" class="run-btn">상환 실행</button>
      </form>
      <div id="repay-result" class="critic-test-result"></div>
    </section>

    <section class="card">
      <div class="card-head">
        <div class="card-title">하드 캡 테스트</div>
        <div class="card-note">지갑 발급·devnet 송금 없이 건별 한도 체크만 실행 — 통제된 자금(Controlled Funds) 증명 (진행 키 필요)</div>
      </div>
      <form id="hardcap-test-form" class="demo-form">
        <select name="scenario" id="hardcap-scenario-select">
          <option value="violation">위반 시나리오 — 6,000,000원 요청 (건별 한도 500만원 초과)</option>
          <option value="clean">정상 시나리오 — 100,000원 요청 (한도 이내)</option>
        </select>
        <button type="submit" class="run-btn">테스트 실행</button>
      </form>
      <div id="hardcap-test-result" class="critic-test-result"></div>
    </section>
  </div>

  <script>
    function getDemoKey() {{
      return prompt('데모 키를 입력하세요');
    }}

    async function refreshLiveRegion() {{
      try {{
        const res = await fetch('/live-region');
        if (res.ok) {{
          const text = await res.text();
          const [top, bottom] = text.split('{_LIVE_REGION_SPLIT}');
          document.getElementById('live-region-top').innerHTML = top;
          document.getElementById('live-region-bottom').innerHTML = bottom || '';
        }}
      }} catch (e) {{ /* 폴링 실패는 조용히 무시하고 다음 주기에 재시도 */ }}
    }}

    function walletAddrShort(addr) {{
      return addr.slice(0, 4) + '…' + addr.slice(-4);
    }}

    function buildUnderwriteNarration(data) {{
      if (data.idempotent_replay) {{
        return '최근 ' + {UNDERWRITE_IDEMPOTENCY_MINUTES} + '분 내 이미 처리된 건입니다 — 이전 결과를 그대로 보여드려요 (' + (data.decision || '-') + ').';
      }}
      if (data.decision === 'reject') {{
        return '거절 — 대출을 집행하지 않았습니다 (지갑 발급도 하지 않습니다).';
      }}
      const decisionLabel = data.decision === 'approve' ? '승인' : '조건부승인';
      const addr = data.wallet_address ? walletAddrShort(data.wallet_address) : '-';
      if (data.wallet_newly_issued) {{
        return decisionLabel + ' — 지갑이 없어 임베디드 지갑을 자동 발급(Passkey 방식, 가스비 없음)하고 즉시 대출을 집행했습니다: ' + addr;
      }}
      return decisionLabel + ' — 기존 지갑으로 대출을 즉시 집행했습니다: ' + addr;
    }}

    document.getElementById('demo-underwrite-form').addEventListener('submit', async function (e) {{
      e.preventDefault();
      const key = getDemoKey();
      if (!key) return;
      const applicantId = document.getElementById('demo-applicant-select').value;
      const statusEl = document.getElementById('demo-status');
      statusEl.textContent = '심사 중 — 에이전트가 정량/정성 정보를 검토하고 있습니다…';
      const body = new URLSearchParams({{applicant_id: applicantId, key: key}});
      try {{
        const res = await fetch('/demo/underwrite', {{method: 'POST', body: body}});
        if (res.status === 403) {{
          statusEl.textContent = '키가 틀렸습니다. 다시 시도해 주세요.';
          return;
        }}
        if (!res.ok) {{
          let detail = '심사 요청 실패';
          try {{
            const errBody = await res.json();
            if (errBody.detail) detail = errBody.detail;
          }} catch (parseErr) {{ /* 본문이 JSON이 아니면 기본 메시지 사용 */ }}
          statusEl.textContent = detail;
          return;
        }}
        const data = await res.json();
        statusEl.textContent = buildUnderwriteNarration(data);
        refreshLiveRegion();
      }} catch (err) {{
        statusEl.textContent = '요청 중 오류가 발생했습니다.';
      }}
    }});

    function escapeHtml(s) {{
      const div = document.createElement('div');
      div.textContent = s;
      return div.innerHTML;
    }}

    document.getElementById('critic-test-form').addEventListener('submit', async function (e) {{
      e.preventDefault();
      const key = getDemoKey();
      if (!key) return;
      const scenario = document.getElementById('critic-scenario-select').value;
      const resultEl = document.getElementById('critic-test-result');
      resultEl.style.display = 'block';
      resultEl.textContent = 'Critic Agent 실행 중…';
      const body = new URLSearchParams({{scenario: scenario, key: key}});
      try {{
        const res = await fetch('/demo/critic-test', {{method: 'POST', body: body}});
        if (res.status === 403) {{
          resultEl.textContent = '키가 틀렸습니다. 다시 시도해 주세요.';
          return;
        }}
        if (!res.ok) {{
          resultEl.textContent = '테스트 실패';
          return;
        }}
        const data = await res.json();
        const badgeCls = data.verdict === 'reject' ? 'badge-reject' : 'badge-approve';
        const label = data.verdict === 'reject' ? '반박 (Reject)' : '승인 (Approve)';
        let out = '<span class="badge ' + badgeCls + '"><span class="badge-dot"></span>Critic 판정: ' + label + '</span>';
        out += '<div style="margin-top:8px; color:var(--ink-2)">' + escapeHtml(data.critique_reasoning) + '</div>';
        if (data.policy_violation) {{
          out += '<div style="margin-top:6px; color:var(--bad); font-weight:600">⚠ 위반 조항: ' + escapeHtml(data.policy_violation) + '</div>';
        }}
        resultEl.innerHTML = out;
      }} catch (err) {{
        resultEl.textContent = '요청 중 오류가 발생했습니다.';
      }}
    }});

    document.getElementById('repay-form').addEventListener('submit', async function (e) {{
      e.preventDefault();
      const key = getDemoKey();
      if (!key) return;
      const applicantId = document.getElementById('repay-applicant-select').value;
      const resultEl = document.getElementById('repay-result');
      resultEl.style.display = 'block';
      resultEl.textContent = '상환 처리 중…';
      const body = new URLSearchParams({{applicant_id: applicantId, key: key}});
      try {{
        const res = await fetch('/demo/repay', {{method: 'POST', body: body}});
        if (res.status === 403) {{
          resultEl.textContent = '키가 틀렸습니다. 다시 시도해 주세요.';
          return;
        }}
        if (!res.ok) {{
          let detail = '상환 처리 실패';
          try {{
            const errBody = await res.json();
            if (errBody.detail) detail = errBody.detail;
          }} catch (parseErr) {{ /* 본문이 JSON이 아니면 기본 메시지 사용 */ }}
          resultEl.textContent = detail;
          return;
        }}
        const data = await res.json();
        let out = '<span class="badge badge-approve"><span class="badge-dot"></span>상환 완료: ' + data.amount_usdc.toFixed(2) + ' USDC</span>';
        if (data.explorer_url) {{
          out += '<div style="margin-top:8px"><a class="txlink" href="' + data.explorer_url + '" target="_blank" rel="noopener">' + walletAddrShort(data.tx_signature) + '</a></div>';
        }}
        resultEl.innerHTML = out;
        refreshLiveRegion();
      }} catch (err) {{
        resultEl.textContent = '요청 중 오류가 발생했습니다.';
      }}
    }});

    document.getElementById('hardcap-test-form').addEventListener('submit', async function (e) {{
      e.preventDefault();
      const key = getDemoKey();
      if (!key) return;
      const scenario = document.getElementById('hardcap-scenario-select').value;
      const resultEl = document.getElementById('hardcap-test-result');
      resultEl.style.display = 'block';
      resultEl.textContent = '하드 캡 체크 중…';
      const body = new URLSearchParams({{scenario: scenario, key: key}});
      try {{
        const res = await fetch('/demo/hard-cap-test', {{method: 'POST', body: body}});
        if (res.status === 403) {{
          resultEl.textContent = '키가 틀렸습니다. 다시 시도해 주세요.';
          return;
        }}
        if (!res.ok) {{
          let detail = '테스트 실패';
          try {{
            const errBody = await res.json();
            if (errBody.detail) detail = errBody.detail;
          }} catch (parseErr) {{ /* 본문이 JSON이 아니면 기본 메시지 사용 */ }}
          resultEl.textContent = detail;
          return;
        }}
        const data = await res.json();
        const badgeCls = data.blocked ? 'badge-reject' : 'badge-approve';
        const label = data.blocked ? '차단됨 (BLOCKED)' : '통과 (PASSED)';
        let out = '<span class="badge ' + badgeCls + '"><span class="badge-dot"></span>' + data.requested_krw.toLocaleString() + '원 요청 — ' + label + '</span>';
        out += '<div style="margin-top:8px; color:var(--ink-2)">' + escapeHtml(data.message) + '</div>';
        resultEl.innerHTML = out;
      }} catch (err) {{
        resultEl.textContent = '요청 중 오류가 발생했습니다.';
      }}
    }});

    refreshLiveRegion();
    setInterval(refreshLiveRegion, 2500);
  </script>
</body>
</html>
"""
