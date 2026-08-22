"""BigQuery 4개 테이블(loan_decisions/receipts/reevaluations/repayments)에 대한
읽기 전용 데이터 품질 체크.

NULL 체크, 중복 체크, 범위/정합성 체크 세 종류를 SELECT 쿼리로만 수행한다 — 어떤 행도
수정·삭제하지 않는다. 판정 로직(payment_mock.py)이 삽입 시점에 강제하는 하드캡/정책 제약이
운영 중 실제로 깨진 적이 없는지, 사람이 주기적으로 다시 조회해 확인할 수 있게 하는
2차 검증(감사) 도구다 — 하드캡을 집행 계층에서 심사 로직과 독립적으로 한 번 더 강제하는
것과 같은 발상을, 저장된 데이터에 대해 사후적으로 적용한다.

실행:
    cd scripts/agent
    python data_quality_check.py

종료 코드: 이상 없으면 0, 하나 이상 발견되면 1 (CI/스케줄러에서 그대로 활용 가능).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = AGENT_DIR.parent
for _p in (SCRIPTS_DIR, AGENT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bigquery_logger import (  # noqa: E402
    DATASET_ID,
    PROJECT_ID,
    RECEIPTS_TABLE_ID,
    REEVAL_TABLE_ID,
    REPAYMENTS_TABLE_ID,
    TABLE_ID,
    get_client,
)
from payment_mock import DAILY_HARD_CAP_KRW, PER_TX_HARD_CAP_KRW  # noqa: E402


@dataclass
class Violation:
    table: str
    check: str
    detail: str
    count: int
    sample: list = field(default_factory=list)


def _fq(table_id: str) -> str:
    return f"`{PROJECT_ID}.{DATASET_ID}.{table_id}`"


def _run(client, query: str) -> list[dict]:
    return [dict(row) for row in client.query(query).result()]


# 스키마상 NULLABLE이지만, 심사 근거 문서화 의무(정책 4항)상 실무적으로는 항상 채워져야 하는 컬럼들.
NULL_TARGETS = {
    TABLE_ID: ["decision", "status", "rationale", "rationale_hash", "critic_verdict"],
    RECEIPTS_TABLE_ID: ["decision", "status", "tx_signature"],
    REEVAL_TABLE_ID: ["original_decision", "new_decision", "rationale_hash"],
    REPAYMENTS_TABLE_ID: ["status", "rationale_hash"],
}


def check_nulls(client) -> list[Violation]:
    violations = []
    for table_id, cols in NULL_TARGETS.items():
        for col in cols:
            rows = _run(
                client,
                f"SELECT COUNT(*) AS cnt FROM {_fq(table_id)} WHERE {col} IS NULL",
            )
            cnt = rows[0]["cnt"]
            if cnt:
                violations.append(Violation(table_id, "NULL", f"{col} IS NULL", cnt))
    return violations


def check_duplicates(client) -> list[Violation]:
    """tx_signature는 온체인 트랜잭션 1건당 1개여야 한다 — 중복은 이중 기록/재시도 버그 신호.
    reevaluations는 신청자당 최대 1건이어야 한다(bigquery_logger.already_reevaluated의 전제)."""
    violations = []
    for table_id in (TABLE_ID, RECEIPTS_TABLE_ID, REEVAL_TABLE_ID, REPAYMENTS_TABLE_ID):
        rows = _run(
            client,
            f"""
            SELECT tx_signature, COUNT(*) AS cnt
            FROM {_fq(table_id)}
            WHERE tx_signature IS NOT NULL
            GROUP BY tx_signature
            HAVING cnt > 1
            """,
        )
        if rows:
            violations.append(
                Violation(table_id, "DUPLICATE_TX", "tx_signature 중복", sum(r["cnt"] for r in rows), rows[:5])
            )

    rows = _run(
        client,
        f"""
        SELECT applicant_id, COUNT(*) AS cnt
        FROM {_fq(REEVAL_TABLE_ID)}
        GROUP BY applicant_id
        HAVING cnt > 1
        """,
    )
    if rows:
        violations.append(
            Violation(REEVAL_TABLE_ID, "DUPLICATE_REEVAL", "신청자당 재심사 2건 이상", sum(r["cnt"] for r in rows), rows[:5])
        )
    return violations


def check_ranges(client) -> list[Violation]:
    """정책 3/6항 하드캡, 판정-집행 정합성, 상식적 범위(음수 금액)를 벗어난 값 탐지."""
    violations = []

    rows = _run(
        client,
        f"""
        SELECT applicant_id, requested_loan_krw, timestamp
        FROM {_fq(TABLE_ID)}
        WHERE status = 'EXECUTED' AND requested_loan_krw > {PER_TX_HARD_CAP_KRW}
        """,
    )
    if rows:
        violations.append(
            Violation(TABLE_ID, "PER_TX_CAP", f"건별 하드캡({PER_TX_HARD_CAP_KRW:,}원) 초과 집행", len(rows), rows[:5])
        )

    rows = _run(
        client,
        f"""
        SELECT DATE(timestamp) AS disbursed_date, SUM(requested_loan_krw) AS total_krw
        FROM {_fq(TABLE_ID)}
        WHERE status = 'EXECUTED'
        GROUP BY disbursed_date
        HAVING total_krw > {DAILY_HARD_CAP_KRW}
        """,
    )
    if rows:
        violations.append(
            Violation(TABLE_ID, "DAILY_CAP", f"일별 하드캡({DAILY_HARD_CAP_KRW:,}원) 초과", len(rows), rows[:5])
        )

    rows = _run(
        client,
        f"""
        SELECT applicant_id, decision, status, timestamp
        FROM {_fq(TABLE_ID)}
        WHERE decision = 'reject' AND status = 'EXECUTED'
        """,
    )
    if rows:
        violations.append(Violation(TABLE_ID, "REJECT_EXECUTED", "거절 건인데 집행 상태(정책 위반)", len(rows), rows[:5]))

    rows = _run(
        client,
        f"""
        SELECT applicant_id, decision, status, timestamp
        FROM {_fq(TABLE_ID)}
        WHERE status = 'EXECUTED' AND tx_signature IS NULL
        """,
    )
    if rows:
        violations.append(Violation(TABLE_ID, "EXECUTED_NO_TX", "집행 상태인데 tx_signature 없음", len(rows), rows[:5]))

    for table_id, amount_col in ((REPAYMENTS_TABLE_ID, "amount_usdc"), (REEVAL_TABLE_ID, "additional_amount")):
        rows = _run(
            client,
            f"SELECT applicant_id, {amount_col} FROM {_fq(table_id)} WHERE {amount_col} < 0",
        )
        if rows:
            violations.append(Violation(table_id, "NEGATIVE_AMOUNT", f"{amount_col} 음수", len(rows), rows[:5]))

    return violations


def run_all_checks() -> list[Violation]:
    client = get_client()
    return check_nulls(client) + check_duplicates(client) + check_ranges(client)


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        # Windows 콘솔 기본 코드페이지(cp949)에서 em dash 등 일부 문자가 인코딩 실패로
        # 크래시를 일으키는 것을 방지한다.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    violations = run_all_checks()
    if not violations:
        print("이상 없음 — NULL/중복/범위 체크 모두 통과.")
        return 0

    print(f"이상 {len(violations)}건 발견:\n")
    for v in violations:
        print(f"[{v.table}] {v.check}: {v.detail} — {v.count}건")
        for row in v.sample:
            print(f"    {row}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
