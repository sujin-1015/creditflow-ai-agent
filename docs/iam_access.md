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

프로젝트 전체에 `roles/editor`를 여전히 보유하고 있으나, **2026-08-22부로 Cloud Run
런타임에서는 더 이상 쓰이지 않는다**(위 신설 계정으로 교체됨). 이 계정의 `roles/editor`를
회수하는 것은 별도 검토 대상으로 남겨두었다 — 회수 전에 프로젝트 내 다른 리소스가 이 계정에
의존하는지 전수 확인이 필요하기 때문이다(예: Cloud Build는 별도 전용 계정
`46585987317@cloudbuild.gserviceaccount.com`을 쓰는 것으로 확인했지만, 다른 잠재적 의존
관계까지 검증되지는 않았다).

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

전환 직후 다음을 확인했다:

- `gcloud run services describe`로 서비스가 신규 계정으로 실행 중임을 확인.
- `GET /health` → `200 {"status":"ok"}`.
- `GET /`(대시보드, BigQuery/Firestore 조회를 거쳐 렌더링) → `200`, 응답 본문에 권한 오류
  관련 문자열(permission/denied/403/forbidden 등) 없음 — 읽기 경로에서 신규 계정의
  BigQuery/Firestore 권한이 실제로 동작함을 확인.
- `POST /underwrite/{id}` 등 실제 대출 집행을 유발하는 쓰기 경로는 라이브 데모에 실제
  devnet 트랜잭션과 BigQuery 행을 만들기 때문에 이번 검증에서는 실행하지 않았다 — 필요하면
  별도로 승인받아 진행한다.

## 남은 항목 (후속 검토 대상, 이번 작업 범위 밖)

- 기본 컴퓨트 서비스 계정의 `roles/editor` 회수 여부
- 사람 계정(`sujin1015777@gmail.com`)이 보유한 `roles/owner`를 상시 업무에는 더 좁은
  역할로 낮추고, 필요할 때만 상승시키는 구조로 바꿀지 여부
- Cloud Scheduler/Eventarc 등 GCP 관리형 서비스 에이전트 권한의 별도 감사
