"""판정 근거 해시가 새겨진 온체인 메모를 BigQuery 기록과 대조해 위변조 여부를 검증한다.

payment_mock.py는 자금을 집행할 때마다 sha256(rationale)을 SPL Memo 프로그램 명령으로
같은 트랜잭션에 실어 devnet에 함께 새긴다 (devnet_transfer.py의 transfer_usdc memo 인자).
이 스크립트는 반대 방향을 확인한다 — "지금 BigQuery에 저장된 판정 근거가, 그 판정으로
실제 집행된 devnet 트랜잭션에 새겨진 해시와 정확히 일치하는가?" 둘 중 하나라도 사후에
바뀌면(예: BigQuery rationale 컬럼을 누가 조작) 불일치가 드러난다 — 사람이 아니라
온체인 데이터가 판정 근거의 신뢰 앵커가 된다.
"""

import hashlib
import sys
from pathlib import Path

from google.cloud import bigquery

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import devnet_transfer  # noqa: E402
import bigquery_logger  # noqa: E402


def _rows_with_tx(applicant_id: int | None = None) -> list[dict]:
    """tx_signature가 있는 모든 집행 기록을 loan_decisions + reevaluations에서 모아온다."""
    client = bigquery_logger.get_client()
    bigquery_logger.ensure_table()
    bigquery_logger.ensure_reeval_table()

    where_clause = "AND applicant_id = @applicant_id" if applicant_id is not None else ""
    query = f"""
        SELECT applicant_id, decision AS record_decision, rationale, rationale_hash,
               tx_signature, timestamp AS ts, 'loan_decisions' AS source_table
        FROM `{bigquery_logger.PROJECT_ID}.{bigquery_logger.DATASET_ID}.{bigquery_logger.TABLE_ID}`
        WHERE tx_signature IS NOT NULL {where_clause}
        UNION ALL
        SELECT applicant_id, new_decision AS record_decision, rationale, rationale_hash,
               tx_signature, reevaluated_at AS ts, 'reevaluations' AS source_table
        FROM `{bigquery_logger.PROJECT_ID}.{bigquery_logger.DATASET_ID}.{bigquery_logger.REEVAL_TABLE_ID}`
        WHERE tx_signature IS NOT NULL {where_clause}
        ORDER BY ts DESC
    """
    params = []
    if applicant_id is not None:
        params.append(bigquery.ScalarQueryParameter("applicant_id", "INT64", applicant_id))
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return [dict(row) for row in client.query(query, job_config=job_config).result()]


def verify_record(row: dict) -> dict:
    """한 건의 rationale-해시-온체인메모 일치 여부를 확인한다."""
    recomputed_hash = hashlib.sha256(row["rationale"].encode("utf-8")).hexdigest()
    onchain_memo = devnet_transfer.fetch_memo(row["tx_signature"])

    bigquery_hash_ok = recomputed_hash == row["rationale_hash"]
    onchain_hash_ok = bool(onchain_memo) and recomputed_hash in onchain_memo

    return {
        "applicant_id": row["applicant_id"],
        "source_table": row["source_table"],
        "tx_signature": row["tx_signature"],
        "recomputed_hash": recomputed_hash,
        "bigquery_rationale_hash": row["rationale_hash"],
        "onchain_memo": onchain_memo,
        "bigquery_hash_matches": bigquery_hash_ok,
        "onchain_hash_matches": onchain_hash_ok,
        "verified": bigquery_hash_ok and onchain_hash_ok,
    }


def verify_applicant(applicant_id: int) -> list[dict]:
    rows = _rows_with_tx(applicant_id=applicant_id)
    return [verify_record(row) for row in rows]


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = _rows_with_tx(applicant_id=target)
    if not rows:
        print("tx_signature가 있는 기록이 없습니다.")
        sys.exit(0)

    for row in rows:
        result = verify_record(row)
        print("=" * 70)
        print(f"[{result['source_table']}] 신청자 {result['applicant_id']} / tx {result['tx_signature']}")
        print(f"  BigQuery rationale -> sha256   : {result['recomputed_hash']}")
        print(f"  BigQuery 저장된 rationale_hash : {result['bigquery_rationale_hash']}")
        print(f"  온체인 메모                    : {result['onchain_memo']}")
        print(f"  BigQuery 해시 일치              : {result['bigquery_hash_matches']}")
        print(f"  온체인 메모 해시 일치           : {result['onchain_hash_matches']}")
        print(f"  ==> 검증 결과: {'✅ 위변조 없음' if result['verified'] else '❌ 불일치 발견'}")
