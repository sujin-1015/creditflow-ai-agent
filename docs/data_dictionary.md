# 데이터 딕셔너리

CreditFlow AI Agent가 다루는 모든 데이터 자산(원본 → 전처리 → 모델 → 판정 → 온체인/BigQuery/Firestore)의
필드 정의를 한 곳에 모은 문서. 데이터가 어디서 생성되고 무엇을 의미하는지 심사역·개발자·감사자가
코드를 읽지 않고도 파악할 수 있게 하는 것이 목적이다.

각 표의 **원천(Source)** 컬럼은 이 값이 어디서 오는지를 가리키며, 흐름 자체는
[data_lineage.md](data_lineage.md)에서 별도로 다룬다.

---

## 1. 원본 데이터 (Kaggle "Loan Prediction Based on Customer Behavior")

용량 문제로 저장소에는 포함되지 않으며, 실행 전 프로젝트 루트에 직접 배치해야 한다
(`Training Data.csv`, `Test Data.csv`, `Sample Prediction Dataset.csv`). **읽기 전용** —
`scripts/preprocess.py`는 이 파일을 절대 수정하지 않는다.

| 원본 컬럼 | 전처리 후 컬럼명 | 타입 | 의미 |
|---|---|---|---|
| `Id` | `id` | INTEGER | 신청자 고유 ID |
| `Income` | `annual_revenue_krw` | INTEGER | 연매출(원). 컬럼명은 KRW 기준이지만 원본 값은 통화 단위가 명시되지 않은 Kaggle 데이터를 그대로 사용 |
| `Age` | `age` | INTEGER | 신청자 나이 |
| `Experience` | `career_years` | INTEGER | 총 경력 연수 |
| `Married/Single` | `household_status` → `is_married`(0/1) | STRING → INT | 혼인 여부 |
| `House_Ownership` | `biz_premise_ownership` → 원-핫 3열 | STRING → INT | 사업장(주거) 소유 형태: owned/rented/norent_noown |
| `Car_Ownership` | `has_business_asset_car`(has_car, 0/1) | STRING → INT | 차량(사업용 자산으로 간주) 보유 여부 |
| `Profession` | `industry_sector` → `industry_sector_risk_te` | STRING → FLOAT | 업종. 고카디널리티라 target encoding으로 변환 |
| `CITY` | `biz_city` → `biz_city_risk_te` | STRING → FLOAT | 사업장 소재 도시. 위키백과 각주 잔재(`[10]` 등) 제거 후 사용 |
| `STATE` | `biz_region` → `biz_region_risk_te` | STRING → FLOAT | 사업장 소재 주(州)/지역 |
| `CURRENT_JOB_YRS` | `biz_operation_years` | INTEGER | 현 사업 운영 연차 |
| `CURRENT_HOUSE_YRS` | `biz_location_years` | INTEGER | 현 사업장 소재지 거주/입주 연차 |
| `Risk_Flag` | `default_risk_label` | INTEGER(0/1) | 실제 부도 여부(라벨). **`Training Data.csv`에만 존재** — `Test Data.csv`는 라벨이 없어 학습/검증에 쓸 수 없다 |

> Kaggle 원본에는 매출 변동성·부채비율에 해당하는 컬럼이 없다. 이 두 값은 전처리 단계에서
> PoC용으로 합성하며, 아래 "전처리 피처" 표의 `revenue_volatility_synth`/`debt_to_income_synth`를 참고.

## 2. 전처리 산출물 (`processed/`)

`scripts/preprocess.py` 실행 결과. 컬럼 정의는 `processed/feature_manifest.md`가 1차 소스이며,
아래는 이를 데이터 사전 형식으로 재정리한 것이다.

| 파일 | 내용 |
|---|---|
| `train.csv` / `val.csv` / `test.csv` | `id` + 피처 18개 + `default_risk_label`. `default_risk_label` 기준 stratified split (seed=42), 비율 70/15/15 |
| `encoding_maps.json` | `industry_sector`/`biz_city`/`biz_region`의 target encoding 매핑(카테고리 → 부도율, 스무딩 20) + `global_mean`. train에서만 학습, val/test는 train 매핑을 그대로 적용(리키지 방지) |
| `feature_manifest.md` | 분할 통계, 피처 목록, 주의사항 — 전처리 스크립트가 실행마다 자동 재생성 |

### 피처 컬럼 (18개)

| 컬럼 | 타입 | 원천 | 의미 |
|---|---|---|---|
| `annual_revenue_krw` | FLOAT | 원본 `Income` | 연매출 |
| `income_log` | FLOAT | 파생(`log1p(annual_revenue_krw)`) | 매출 로그 변환 |
| `income_per_age` | FLOAT | 파생(`annual_revenue_krw / age`) | 연령 대비 매출 |
| `age` | INTEGER | 원본 `Age` | 나이 |
| `career_years` | INTEGER | 원본 `Experience` | 총 경력 연수 |
| `biz_operation_years` | INTEGER | 원본 `CURRENT_JOB_YRS` | 현 사업 운영 연차 |
| `biz_location_years` | INTEGER | 원본 `CURRENT_HOUSE_YRS` | 현 사업장 소재지 연차 |
| `job_stability_ratio` | FLOAT | 파생(`biz_operation_years / career_years`) | 업력 안정성(0으로 나누기 방지 처리) |
| `is_married` | INT(0/1) | 원본 `Married/Single` | 혼인 여부 |
| `has_car` | INT(0/1) | 원본 `Car_Ownership` | 차량 보유 여부 |
| `revenue_volatility_synth` | FLOAT | **합성(synthetic)** — id 시드 고정 난수, 범위 0.05~0.6 | 매출 변동성 PoC 대체값. 원본에 없음, 타겟과 무관한 순수 노이즈. 실 서비스 전환 시 계좌/매출 API로 교체 필요 |
| `debt_to_income_synth` | FLOAT | **합성(synthetic)** — id 시드 고정 난수, 범위 0.1~0.9 | 부채비율 PoC 대체값. 위와 동일한 성격 |
| `industry_sector_risk_te` | FLOAT | 파생(target encoding of `industry_sector`) | 업종별 위험도(부도율 기반) |
| `biz_city_risk_te` | FLOAT | 파생(target encoding of `biz_city`) | 도시별 위험도 |
| `biz_region_risk_te` | FLOAT | 파생(target encoding of `biz_region`) | 지역별 위험도 |
| `biz_premise_ownership_owned` | INT(0/1) | 원본 `House_Ownership` 원-핫 | 사업장 자가 소유 |
| `biz_premise_ownership_rented` | INT(0/1) | 원본 `House_Ownership` 원-핫 | 사업장 임차 |
| `biz_premise_ownership_norent_noown` | INT(0/1) | 원본 `House_Ownership` 원-핫 | 무임차·무소유 |

> `revenue_volatility_synth`/`debt_to_income_synth`는 모델 성능 지표 해석 시 기여도를 과신하면 안 되는
> PoC 플레이스홀더다 — [processed/feature_manifest.md](../processed/feature_manifest.md) 참고.

## 3. 모델 아티팩트 (`models/`)

| 파일 | 내용 |
|---|---|
| `xgboost_model.joblib` | 실제 서빙에 쓰이는 모델(`data_tools.predict_risk`가 로드) |
| `lightgbm_model.joblib` | 비교용으로만 학습, 서빙에는 미포함 |
| `threshold_config.json` | `t_approve`(0.5877), `t_reject`(0.6264) — validation set에서 승인군 부도율 8% 이하·거절군 20% 이상 제약을 만족하도록 산정. `approve_bad_rate`/`reject_bad_rate`/`constraint_used`에 근거 수치 기록 |
| `metrics_report.json` | Test AUC 등 전체 성능 지표 |
| `kfold_cv_report.json` | 5-fold 교차검증 결과(AUC 재현성 확인) |
| `model_report.md` | 위 지표를 사람이 읽을 수 있게 정리한 리포트 |

`quant_tier`(정량 등급) 판정 규칙: `default_probability < t_approve` → approve,
`>= t_reject` → reject, 그 사이 → conditional (`scripts/agent/data_tools.py:predict_risk`).

## 4. BigQuery — `gc-hackathon-504210.creditflow_agent`

판정·집행·재심사·상환의 모든 결과가 최종적으로 적재되는 테이블. 스키마 정의는
`scripts/agent/bigquery_logger.py`가 유일한 소스이며, 아래는 그 내용을 문서화한 것이다.
(GCP 리소스 이름 규칙상 컬럼명은 스키마 정의와 동일하게 snake_case 영문으로 유지한다.)

### 4.1 `loan_decisions` — 1차 심사 판정 + devnet 집행 기록

| 컬럼 | 타입 | 필수 | 의미 |
|---|---|---|---|
| `applicant_id` | INTEGER | ✅ | 신청자 ID |
| `decision` | STRING | ✅ | 최종 판정: `approve` / `conditional` / `reject` |
| `wallet_address` | STRING | | 신청자 Solana devnet 지갑 주소 (없으면 승인 시 자동 발급) |
| `requested_loan_krw` | INTEGER | | 산정된 대출 금액(원) |
| `devnet_test_amount_sol` | FLOAT | | **레거시** — SOL 정산 시절 기록, 하위호환용 (신규 로직 미사용) |
| `devnet_test_amount` | FLOAT | | USDC 전환 이후 실제 집행 금액 |
| `currency` | STRING | | 통화 단위 (예: `USDC`) |
| `status` | STRING | | 집행 상태: `EXECUTED` / `SKIPPED` 등 |
| `tx_signature` | STRING | | Solana 트랜잭션 서명. 미집행(거절) 건은 null |
| `explorer_url` | STRING | | 트랜잭션 탐색기 URL |
| `network` | STRING | | 예: `solana-devnet-mock` |
| `is_mock` | BOOLEAN | | mock 집행 여부 |
| `timestamp` | TIMESTAMP | ✅ | 판정/집행 시각 |
| `rationale` | STRING | | 최종 판정 근거(자연어) |
| `rationale_hash` | STRING | | `sha256(rationale)` — 온체인 SPL Memo와 대조해 위변조 검증 |
| `critic_verdict` | STRING | | Critic Agent 독립 재검토 결과: `approve` \| `reject` |
| `critic_reasoning` | STRING | | Critic Agent 재검토 사유 |
| `tool_call_summary` | STRING | | Decision Agent가 자율적으로 고른 도구 호출 순서 (예: `predict_risk → record_decision`) |

### 4.2 `receipts` — 영수증 (Eventarc + Cloud Workflows 파이프라인 전용, `loan_decisions`와 분리)

| 컬럼 | 타입 | 필수 | 의미 |
|---|---|---|---|
| `applicant_id` | INTEGER | ✅ | 신청자 ID |
| `decision` | STRING | | 판정 결과 |
| `status` | STRING | | 집행 상태 |
| `tx_signature` | STRING | | 트랜잭션 서명 |
| `explorer_url` | STRING | | 탐색기 URL |
| `approved_amount_krw` | INTEGER | | 승인 금액(원) |
| `receipt_issued_at` | TIMESTAMP | ✅ | 영수증 발행 시각 |

### 4.3 `reevaluations` — 조건부승인 건 자동 재심사 기록

| 컬럼 | 타입 | 필수 | 의미 |
|---|---|---|---|
| `applicant_id` | INTEGER | ✅ | 신청자 ID |
| `original_decision` | STRING | | 최초 판정 |
| `new_decision` | STRING | | 재심사 판정 |
| `upgraded` | BOOLEAN | | 조건부승인 → 승인 상향 여부 |
| `additional_amount` | FLOAT | | 상향 시 추가 집행 금액 |
| `currency` | STRING | | 통화 단위 |
| `tx_signature` | STRING | | 추가 집행 트랜잭션 서명 |
| `explorer_url` | STRING | | 탐색기 URL |
| `rationale` | STRING | | 재심사 판정 근거 |
| `rationale_hash` | STRING | | `sha256(rationale)` |
| `critic_verdict` | STRING | | Critic Agent 재검토 결과 |
| `critic_reasoning` | STRING | | Critic Agent 재검토 사유 |
| `tool_call_summary` | STRING | | 재심사 시 도구 호출 순서(상환 이력 조회 포함 가능) |
| `reevaluated_at` | TIMESTAMP | ✅ | 재심사 시각 |

### 4.4 `repayments` — 온체인 상환 기록 (지급의 역방향: 신청자 → treasury)

| 컬럼 | 타입 | 필수 | 의미 |
|---|---|---|---|
| `applicant_id` | INTEGER | ✅ | 신청자 ID |
| `amount_usdc` | FLOAT | | 상환 금액(USDC) |
| `currency` | STRING | | 통화 단위 |
| `status` | STRING | | 상환 트랜잭션 상태 |
| `tx_signature` | STRING | | 트랜잭션 서명 |
| `explorer_url` | STRING | | 탐색기 URL |
| `network` | STRING | | 예: `solana-devnet-mock` |
| `is_mock` | BOOLEAN | | mock 여부 |
| `timestamp` | TIMESTAMP | ✅ | 상환 시각 |
| `rationale` | STRING | | 상환 관련 메모 |
| `rationale_hash` | STRING | | `sha256(rationale)` |

`get_repayment_history(applicant_id)`가 이 테이블을 조회해 재심사 시 Decision Agent의
`get_repayment_history_tool`에 데이터를 공급한다.

## 5. Firestore — `in_progress_decisions` (대시보드 실시간 상태, 순수 UX 부가 기능)

| 필드 | 타입 | 의미 |
|---|---|---|
| `applicant_id` | INTEGER | 심사 진행 중인 신청자 ID (문서 ID와 동일 값을 문자열로도 보관) |
| `started_at` | TIMESTAMP(`SERVER_TIMESTAMP`) | 심사 시작 시각 |

2분(`STALE_AFTER`) 이상 지난 문서는 다음 조회 시 자동 삭제된다 — 컨테이너 비정상 종료로
`clear_in_progress()`가 실행되지 못해도 "심사 중" 표시가 영구히 남지 않도록 하는 자가 정리 로직
(`scripts/agent/live_state.py`). 실패해도 심사/집행 흐름을 막지 않는 부가 데이터이므로 BigQuery처럼
영구 감사 기록으로 취급하지 않는다.

## 6. 로컬 온체인 로그 파일 (`onchain/`)

BigQuery와 별도로, devnet 데모 실행 시 로컬 파일에도 동일 계열 데이터를 남긴다(발표/디버깅용 미러).

| 파일 | 내용 |
|---|---|
| `payments_log.json` | `loan_decisions`와 거의 동일한 필드(`applicant_id`, `decision`, `wallet_address`, `requested_loan_krw`, `devnet_test_amount_usdc`, `status`, `tx_signature`, `network`, `is_mock`, `timestamp`, `rationale`) — 지급 실행 시마다 append |
| `repayments_log.json` | 상환 실행 로그, `repayments` 테이블과 대응 |
| `devnet_transactions.json` | 저수준 devnet 트랜잭션 기록 (`scripts/devnet_transfer.py`) |
| `devnet_keys/` | devnet 테스트용 지갑 키페어 — **민감 정보**, 실서비스 키가 아니어도 저장소 공유 시 주의 |
| `rationale_reports/` | `scripts/agent/report.py`가 생성하는 신청자별 근거 리포트 |

> `payments_log.json`의 `rationale` 필드가 일부 행에서 한글이 깨져(mojibake) 보이는 건 로그 생성
> 당시 인코딩 이슈이며 데이터 자체(판정 결과, 금액, 트랜잭션 서명)의 정확성과는 무관하다.
