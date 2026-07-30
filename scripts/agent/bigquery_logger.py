"""판정결과 + devnet 트랜잭션 해시를 BigQuery에 저장한다 (검증 가능한 기록).

사전 요구사항: `gcloud auth application-default login` 완료, BigQuery API 활성화
(둘 다 이 세션에서 확인/완료됨).
"""

from datetime import datetime, timezone

from google.cloud import bigquery

PROJECT_ID = "fundbridge-hackathon"
DATASET_ID = "creditflow_agent"
TABLE_ID = "loan_decisions"
LOCATION = "asia-northeast3"

SCHEMA = [
    bigquery.SchemaField("applicant_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("decision", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("wallet_address", "STRING"),
    bigquery.SchemaField("requested_loan_krw", "INTEGER"),
    bigquery.SchemaField("devnet_test_amount_sol", "FLOAT"),  # 레거시 (SOL 정산 시절 기록, 하위호환용)
    bigquery.SchemaField("devnet_test_amount", "FLOAT"),  # USDC 전환 이후 사용하는 필드
    bigquery.SchemaField("currency", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("tx_signature", "STRING"),
    bigquery.SchemaField("explorer_url", "STRING"),
    bigquery.SchemaField("network", "STRING"),
    bigquery.SchemaField("is_mock", "BOOLEAN"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("rationale", "STRING"),
    bigquery.SchemaField("rationale_hash", "STRING"),  # sha256(rationale) — 결제 tx의 온체인 메모와 대조해 위변조 검증
]

_client = None


def get_client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT_ID)
    return _client


def ensure_table() -> bigquery.Table:
    client = get_client()
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)

    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = LOCATION
        client.create_dataset(dataset)
        print(f"BigQuery 데이터셋 생성: {PROJECT_ID}.{DATASET_ID}")

    table_ref = dataset_ref.table(TABLE_ID)
    try:
        table = client.get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=SCHEMA)
        table = client.create_table(table)
        print(f"BigQuery 테이블 생성: {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")
        return table

    # 기존 테이블에 새 컬럼(devnet_test_amount, currency 등)이 없으면 추가한다 (NULLABLE만 추가 가능).
    existing_names = {f.name for f in table.schema}
    missing = [f for f in SCHEMA if f.name not in existing_names]
    if missing:
        table.schema = list(table.schema) + missing
        table = client.update_table(table, ["schema"])
        print(f"BigQuery 테이블 스키마 갱신: {[f.name for f in missing]} 추가")
    return table


RECEIPTS_TABLE_ID = "receipts"

RECEIPTS_SCHEMA = [
    bigquery.SchemaField("applicant_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("decision", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("tx_signature", "STRING"),
    bigquery.SchemaField("explorer_url", "STRING"),
    bigquery.SchemaField("approved_amount_krw", "INTEGER"),
    bigquery.SchemaField("receipt_issued_at", "TIMESTAMP", mode="REQUIRED"),
]


def ensure_receipts_table() -> bigquery.Table:
    """Eventarc+Workflows 파이프라인이 쓰는 영수증 테이블 (loan_decisions와 분리된 별도 테이블)."""
    client = get_client()
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    table_ref = dataset_ref.table(RECEIPTS_TABLE_ID)
    try:
        return client.get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=RECEIPTS_SCHEMA)
        table = client.create_table(table)
        print(f"BigQuery 테이블 생성: {PROJECT_ID}.{DATASET_ID}.{RECEIPTS_TABLE_ID}")
        return table


REEVAL_TABLE_ID = "reevaluations"

REEVAL_SCHEMA = [
    bigquery.SchemaField("applicant_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("original_decision", "STRING"),
    bigquery.SchemaField("new_decision", "STRING"),
    bigquery.SchemaField("upgraded", "BOOLEAN"),
    bigquery.SchemaField("additional_amount", "FLOAT"),
    bigquery.SchemaField("currency", "STRING"),
    bigquery.SchemaField("tx_signature", "STRING"),
    bigquery.SchemaField("explorer_url", "STRING"),
    bigquery.SchemaField("rationale", "STRING"),
    bigquery.SchemaField("rationale_hash", "STRING"),
    bigquery.SchemaField("reevaluated_at", "TIMESTAMP", mode="REQUIRED"),
]


def ensure_reeval_table() -> bigquery.Table:
    """3개월 후 자동 재심사(조건부->승인 상향 여부) 기록 테이블."""
    client = get_client()
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    table_ref = dataset_ref.table(REEVAL_TABLE_ID)
    try:
        table = client.get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=REEVAL_SCHEMA)
        table = client.create_table(table)
        print(f"BigQuery 테이블 생성: {PROJECT_ID}.{DATASET_ID}.{REEVAL_TABLE_ID}")
        return table

    existing_names = {f.name for f in table.schema}
    missing = [f for f in REEVAL_SCHEMA if f.name not in existing_names]
    if missing:
        table.schema = list(table.schema) + missing
        table = client.update_table(table, ["schema"])
        print(f"BigQuery 테이블 스키마 갱신({REEVAL_TABLE_ID}): {[f.name for f in missing]} 추가")
    return table


def log_reevaluation(record: dict) -> None:
    table = ensure_reeval_table()
    client = get_client()
    row = {
        "applicant_id": record["applicant_id"],
        "original_decision": record.get("original_decision"),
        "new_decision": record.get("new_decision"),
        "upgraded": record.get("upgraded"),
        "additional_amount": record.get("additional_amount"),
        "currency": record.get("currency"),
        "tx_signature": record.get("tx_signature"),
        "explorer_url": record.get("explorer_url"),
        "rationale": record.get("rationale"),
        "rationale_hash": record.get("rationale_hash"),
        "reevaluated_at": record.get("reevaluated_at") or datetime.now(timezone.utc).isoformat(),
    }
    errors = client.insert_rows_json(table, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert 실패: {errors}")


def already_reevaluated(applicant_id: int) -> bool:
    ensure_reeval_table()
    client = get_client()
    query = (
        f"SELECT COUNT(*) as cnt FROM `{PROJECT_ID}.{DATASET_ID}.{REEVAL_TABLE_ID}` "
        f"WHERE applicant_id = @applicant_id"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("applicant_id", "INT64", applicant_id)]
    )
    rows = list(client.query(query, job_config=job_config).result())
    return rows[0]["cnt"] > 0


def find_due_conditionals(min_days: int = 90) -> list[dict]:
    """min_days 이상 지난 conditional 판정 중 아직 재심사 안 한 건을 찾는다."""
    ensure_table()
    ensure_reeval_table()
    client = get_client()
    query = f"""
        SELECT d.applicant_id, d.timestamp, d.wallet_address, d.rationale
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` d
        WHERE d.decision = 'conditional'
          AND d.status = 'EXECUTED'
          AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), d.timestamp, DAY) >= @min_days
          AND d.applicant_id NOT IN (
              SELECT applicant_id FROM `{PROJECT_ID}.{DATASET_ID}.{REEVAL_TABLE_ID}`
          )
        QUALIFY ROW_NUMBER() OVER (PARTITION BY d.applicant_id ORDER BY d.timestamp DESC) = 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("min_days", "INT64", min_days)]
    )
    return [dict(row) for row in client.query(query, job_config=job_config).result()]


def log_decision(record: dict) -> None:
    """payment_mock.PaymentResult(dict 형태)를 BigQuery에 한 행 적재한다."""
    table = ensure_table()
    client = get_client()

    row = {
        "applicant_id": record["applicant_id"],
        "decision": record["decision"],
        "wallet_address": record.get("wallet_address"),
        "requested_loan_krw": record.get("requested_loan_krw"),
        "devnet_test_amount": record.get("devnet_test_amount"),
        "currency": record.get("currency"),
        "status": record.get("status"),
        "tx_signature": record.get("tx_signature"),
        "explorer_url": record.get("explorer_url"),
        "network": record.get("network"),
        "is_mock": record.get("is_mock"),
        "timestamp": record.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "rationale": record.get("rationale"),
        "rationale_hash": record.get("rationale_hash"),
    }

    errors = client.insert_rows_json(table, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert 실패: {errors}")


if __name__ == "__main__":
    ensure_table()
    print("테이블 준비 완료. 조회 쿼리 예시:")
    print(f"  SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` ORDER BY timestamp DESC LIMIT 10")
