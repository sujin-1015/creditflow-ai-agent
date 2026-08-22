# IAM 접근 권한 구조

Cloud Run 런타임이 실제로 어떤 신원(identity)으로, 어떤 리소스에, 어떤 범위의 권한으로
접근하는지 정리한 문서. `gc-hackathon-504210` 프로젝트의 실제 IAM 상태를 반영하며,
변경 시 이 문서도 함께 갱신한다.

---

## 변경 이력

| 일자 | 변경 | 사유 |
|---|---|---|
| 2026-08-22 이전 | Cloud Run이 GCP 기본 컴퓨트 서비스 계정(`46585987317-compute@developer.gserviceaccount.com`)으로 실행, 이 계정은 프로젝트 전체에 `roles/editor`(사실상 대부분의 리소스에 읽기/쓰기 가능) 보유 | 배포 시 기본값을 그대로 사용(해커톤 PoC 초기 세팅) |
| 2026-08-22 | 전용 최소 권한 서비스 계정 `creditflow-agent-run@gc-hackathon-504210.iam.gserviceaccount.com` 신설, 아래 "현재 구조"의 리소스별 권한만 부여 후 Cloud Run 런타임을 이 계정으로 전환 | 데이터 거버넌스 — 애플리케이션이 실제로 쓰는 범위(BigQuery 1개 데이터셋, Firestore, Pub/Sub 1개 토픽, Secret Manager 시크릿 1개)를 넘어서는 프로젝트 전체 Editor 권한으로 운영되고 있던 것을 최소 권한 원칙에 맞게 좁힘 |
| 2026-08-22 | 기본 컴퓨트 서비스 계정에 `roles/eventarc.eventReceiver`(프로젝트), `roles/workflows.invoker`(프로젝트 — Workflows는 리소스 단위 IAM 바인딩 명령이 없어 프로젝트 단위로만 부여 가능, 현재 워크플로우가 1개뿐이라 실질 범위는 동일), BigQuery `creditflow_agent` 데이터셋 `WRITER`(classic ACL)를 명시적으로 부여한 뒤 `roles/editor`를 회수 | Cloud Run 이관 후 재조사 결과 기본 컴퓨트 계정이 Eventarc 트리거 `payment-events-trigger`의 신원, Cloud Workflow `payment-receipt-workflow`의 실행 신원으로도 여전히 쓰이고 있어(README의 `PS → EA → WF → BQ receipts` 경로) Editor의 광범위한 권한에 암묵적으로 의존하고 있었음. Editor 없이도 동작하도록 필요한 권한만 먼저 명시적으로 부여한 뒤 Editor를 제거 |

## 현재 구조

### `creditflow-agent-run@gc-hackathon-504210.iam.gserviceaccount.com` — Cloud Run 런타임 (신설)

`service/main.py`(FastAPI, Cloud Run에서 실행)와 그 안에서 호출되는 `scripts/agent/*.py`가
실제로 필요로 하는 범위만 부여했다.

| 리소스 | 역할 | 범위 | 부여 방식 |
|---|---|---|---|
| BigQuery 데이터셋 `creditflow_agent` | `WRITER`(dataEditor 상당) | 데이터셋 1개 (프로젝트 전체 아님) | 데이터셋 access 목록 (`bq update` — `bq add/get-iam-policy`가 조직 정책상 allowlist 필요해 막혀 있어 classic ACL 방식으로 부여) |
| BigQuery (프로젝트) | `roles/bigquery.jobUser` | 프로젝트 전체 (쿼리 job 실행 권한은 데이터셋 단위로 세분화 불가) | 프로젝트 IAM 바인딩 |
| Firestore | `roles/datastore.user` | 프로젝트 전체 (Firestore는 컬렉션 단위 IAM 미지원, `in_progress_decisions` 하나만 실제 사용) | 프로젝트 IAM 바인딩 |
| Pub/Sub 토픽 `payment-events` | `roles/pubsub.publisher` | 토픽 1개 | 토픽 IAM 바인딩 |
| Secret Manager 시크릿 `gemini-api-key` | `roles/secretmanager.secretAccessor` | 시크릿 1개 | 시크릿 IAM 바인딩 |

이 계정에는 `roles/editor`, `roles/owner` 등 프로젝트 전반에 영향을 주는 광범위한 역할이
전혀 없다 — 위 5개 바인딩이 전부다.

### `46585987317-compute@developer.gserviceaccount.com` — 기본 컴퓨트 서비스 계정

Cloud Run 런타임에서는 더 이상 쓰이지 않지만(위 신설 계정으로 교체됨), 조사 결과 아래 두
곳에서 여전히 실제로 쓰이고 있어 **완전히 걷어내지는 않고, `roles/editor`만 회수하고
필요한 권한을 명시적으로 남겼다**.

| 사용처 | 용도 | 남긴 권한 | 범위 |
|---|---|---|---|
| Eventarc 트리거 `payment-events-trigger` | `payment-events` Pub/Sub 이벤트를 받아 Workflow를 실행시키는 신원 | `roles/eventarc.eventReceiver` | 프로젝트 |
| Cloud Workflow `payment-receipt-workflow` | 워크플로우 자체의 실행 신원(내부에서 BigQuery `receipts`에 씀) | `roles/workflows.invoker`, BigQuery `creditflow_agent` 데이터셋 `WRITER` | 프로젝트 / 데이터셋 1개 |
| (배포 시 `--set-secrets`로 부여됨, 이번 작업 이전부터 존재) | Cloud Run이 기본 계정으로 실행되던 시절의 잔존 바인딩 | `roles/secretmanager.secretAccessor` (`gemini-api-key`) | 시크릿 1개 |

`roles/editor`는 2026-08-22에 회수했다. Cloud Build는 별도 전용 계정
(`46585987317@cloudbuild.gserviceaccount.com`)을 쓰는 것을 확인했고, Compute Engine/Cloud
Functions/App Engine/GKE는 모두 API 자체가 비활성 상태라(`gcloud compute instances list`
등이 `SERVICE_DISABLED`로 실패) 이 계정에 의존하는 다른 리소스가 없음을 확인한 뒤 진행했다.
Secret Manager 잔존 바인딩은 위험도가 낮아(시크릿 1개, 읽기 전용) 이번엔 정리하지 않았다 —
다음에 이 계정을 다시 손볼 때 함께 정리 대상.

### 기타 GCP 관리형 서비스 에이전트

`*.gserviceaccount.com` 형태의 `roles/*.serviceAgent`류 계정(Cloud Build, Eventarc,
Firestore, Pub/Sub, Cloud Run, Workflows 등)은 GCP가 각 API 활성화 시 자동 생성·관리하는
시스템 계정으로, 이 프로젝트에서 직접 만들거나 권한을 조정한 적이 없다. 이번 작업 대상에서
제외했다.

### Cloud Scheduler → Cloud Run 호출

Cloud Scheduler 잡(`reevaluate-conditional-loans`)은 OIDC 인증 없이 공개 엔드포인트를
호출한다 — Cloud Run 서비스 자체가 `--allow-unauthenticated`로 배포되어 `allUsers`에게
`roles/run.invoker`를 허용하고 있기 때문이다(README의 라이브 데모 URL도 동일하게 공개
접근). 이 공개 접근 설정은 라이브 데모 요구사항이라 이번 IAM 정리 대상에서 제외했다 —
바꾸려면 대시보드 접근 방식 자체를 재설계해야 한다.

## 검증

**Cloud Run 계정 전환 직후**:
- `gcloud run services describe`로 서비스가 신규 계정으로 실행 중임을 확인.
- `GET /health` → `200 {"status":"ok"}`.
- `GET /`(대시보드, BigQuery/Firestore 조회를 거쳐 렌더링) → `200`, 응답 본문에 권한 오류
  관련 문자열(permission/denied/403/forbidden 등) 없음 — 읽기 경로에서 신규 계정의
  BigQuery/Firestore 권한이 실제로 동작함을 확인.

**기본 컴퓨트 계정 Editor 회수 직후**:
- `GET /health` → `200` (Cloud Run 자체는 이미 다른 계정으로 실행 중이라 영향 없음을 재확인).
- Eventarc/Workflow → BigQuery `receipts` 경로는 실제 대출 신청 1건이 승인/조건부승인으로
  집행되어야 이벤트가 발생하는 흐름이라, 이번 세션에서는 트리거하지 않았다 — **아래 "직접
  테스트하는 방법"을 참고해 실제로 검증해보는 것을 권장한다.**
- `POST /underwrite/{id}` 등 실제 대출 집행/온체인 트랜잭션을 유발하는 쓰기 경로는 라이브
  데모에 실제 데이터를 남기므로 이번 작업에서는 실행하지 않았다.

### 직접 테스트하는 방법 (영수증 파이프라인)

Editor 회수가 Eventarc/Workflow 경로를 깨뜨리지 않았는지 직접 확인하려면:

1. 라이브 대시보드(`https://creditflow-agent-46585987317.asia-northeast3.run.app`)에서
   승인 또는 조건부승인이 나올 만한 신청자로 "새 심사 실행"을 돌린다.
2. 몇 초~수십 초 뒤 BigQuery 콘솔에서 다음 쿼리로 해당 `applicant_id`의 영수증 행이
   생겼는지 확인한다:
   ```sql
   SELECT * FROM `gc-hackathon-504210.creditflow_agent.receipts`
   WHERE applicant_id = <신청자ID>
   ORDER BY receipt_issued_at DESC LIMIT 5
   ```
3. 만약 행이 생기지 않으면, Cloud 콘솔의 Workflows → `payment-receipt-workflow` → 실행 기록
   (Executions)에서 실패 사유(주로 권한 오류 메시지)를 확인한다 — 이 경우 위 표의 권한
   중 빠진 것이 있다는 뜻이므로 `docs/iam_access.md`를 다시 참고해 gcloud 명령으로 보강한다.

## 남은 항목 (후속 검토 대상, 이번 작업 범위 밖)

- 기본 컴퓨트 서비스 계정에 남겨둔 Secret Manager `secretAccessor`(`gemini-api-key`) 잔존
  바인딩 정리 — 위험도는 낮지만 더 이상 이 계정이 그 시크릿을 읽을 이유가 없음
- 사람 계정(`sujin1015777@gmail.com`)이 보유한 `roles/owner`를 상시 업무에는 더 좁은
  역할로 낮추고, 필요할 때만 상승시키는 구조로 바꿀지 여부
- Cloud Scheduler/Eventarc 등 GCP 관리형 서비스 에이전트 권한의 별도 감사
