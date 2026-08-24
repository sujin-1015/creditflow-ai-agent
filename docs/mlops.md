# 데이터 파이프라인(ETL)·MLOps

XGBoost 모델의 재학습·버전 관리·배포 게이트를 다루는 문서.
Gemini 기반 판정 파이프라인의 평가 체계는 [genai_quality_evaluation.md](genai_quality_evaluation.md)를 참고.

## 이전 상태의 문제

`train_model.py`는 실행할 때마다 `models/xgboost_model.joblib`을 곧바로 덮어썼다:
- 이전 버전이 언제 학습됐는지, 성능이 얼마였는지 알 방법이 없었다(git 커밋 로그를 뒤지는 것 외에는).
- 재학습한 모델이 기존보다 **더 나빠져도** 그대로 배포됐다 — AUC를 사람이 눈으로 비교하기 전까지는 성능 저하를 막을 방법이 없었다.
- AUC가 비슷해도 예측 확률 분포 자체가 밀렸는지(즉 `threshold_config.json`의 고정 임계값이 여전히 유효한지)는 전혀 확인하지 않았다.

## 구현

### 1. 버전 관리 — [scripts/model_registry.py](../scripts/model_registry.py)

`train_model.py`가 학습한 아티팩트를 이제 곧바로 실서빙 경로에 쓰지 않는다:

1. 후보 모델을 `models/_candidate/`(임시)에 저장
2. test AUC 계산
3. `models/versions/{timestamp}_{git_commit}/`에 후보 스냅샷(모델+임계값+지표) 보관
4. `models/model_registry.json`에 버전 이력 기록(타임스탬프, git 커밋, val/test AUC, 임계값, 승격 여부)
5. 게이트를 통과한 경우에만 실서빙 경로(`models/xgboost_model.joblib`, `threshold_config.json`)를 갱신

`models/versions/`의 바이너리 스냅샷은 `.gitignore`로 제외했다 — 저장소에 남기는 건 작고 사람이
읽을 수 있는 `model_registry.json`(이력)이지, 버전마다 쌓이는 수백 KB 바이너리가 아니다.

### 2. 성능 게이트

`model_registry.evaluate_gate(candidate_test_auc)`가 새 후보의 test AUC를 현재 프로덕션(레지스트리에서
가장 최근에 승격된 버전)과 비교한다. **낮으면 배포하지 않고** 콘솔에 알림만 남긴다 — 레지스트리에는
`promoted: false`로 기록되어 "시도는 했지만 배포되지 않음"이 이력으로 남는다. 첫 버전(비교 대상 없음)은
자동 승격된다.

실제로 더 나쁜 후보를 넣어 게이트가 막는지, 그리고 그 상태에서 실서빙 파일이 정말 안 바뀌는지
바이트 단위로 비교해 검증했다(테스트 후 가짜 기록은 레지스트리에서 제거).

### 3. 드리프트 체크 — [scripts/drift_check.py](../scripts/drift_check.py)

이 프로젝트는 정적 Kaggle 데이터셋 기반 PoC라 "시간에 따라 유입되는 신규 데이터"가 없다 —
그래서 전형적인 "지난달 대비 이번달 신청자 분포" 드리프트 체크는 그대로 적용할 수 없다.

대신 실질적으로 의미 있는 질문으로 바꿨다: **재학습된 후보 모델의 예측 확률 분포가 현재
배포된 모델과 같은 test set에서 얼마나 달라졌는가?** AUC는 비슷해도 확률 분포 자체가 크게
밀리면, `threshold_config.json`에 고정된 t_approve/t_reject가 더 이상 원래 의도한 승인군
부도율 8% 이하 / 거절군 20% 이상 제약을 만족하지 못할 수 있다 — AUC 게이트가 놓칠 수 있는
문제를 이 체크가 보완한다.

PSI(Population Stability Index, 업계 통용 기준)로 측정하며, `train_model.py` 실행 시 게이트
판정 전에 자동으로 실행되어 `models/drift_history.json`에 기록된다.

| PSI | 해석 |
|---|---|
| < 0.1 | 유의미한 변화 없음 |
| 0.1 ~ 0.25 | 중간 수준 변화, 모니터링 권장 |
| >= 0.25 | 큰 변화, threshold 재산정 등 조사 필요 |

합성 데이터로 PSI 계산 자체가 실제로 분포 차이를 감지하는지 검증했다(동일 분포 PSI≈0,
크게 다른 분포 PSI≈4.8 — 임계값 0.25를 훨씬 초과해 정상적으로 플래그됨).

### 파이프라인 실행 순서 (`python scripts/train_model.py`)

```
전처리된 데이터 로드
  → XGBoost/LightGBM 학습, 후보를 models/_candidate/에 저장
  → 드리프트 체크: 후보 vs 현재 프로덕션 확률분포 (PSI)
  → 성능 게이트: 후보 test AUC vs 현재 프로덕션 test AUC
      ├─ 통과 → models/versions/에 스냅샷 + 레지스트리 기록 + 실서빙 경로 갱신
      └─ 차단 → models/versions/에 스냅샷 + 레지스트리 기록(promoted=false)만, 실서빙 경로는 그대로
```

## 하지 못한 것: Cloud Scheduler를 통한 실제 재학습 자동 트리거

원래 요청한 "Cloud Scheduler로 주기적 재학습 트리거"는 현재 아키텍처에서 그대로 구현할 수
없다 — 확인해보니 두 가지 구조적 제약이 있다:

1. **원본 학습 데이터가 배포 컨테이너에 없다.** `Dockerfile`/`.dockerignore`를 보면
   `Training Data.csv`는 명시적으로 제외되어 있고(README에 "용량 문제로 저장소에는 포함하지
   않음"이라고 명시), Cloud Run에는 이미 전처리된 `processed/train.csv` 등만 들어간다.
   `preprocess.py`(원본→피처)는 Cloud Run에서 돌릴 방법이 없다.
2. **Cloud Run은 stateless다.** `train_model.py`(재학습만, 전처리된 데이터로) 자체는 이론상
   Cloud Run에서 돌릴 수 있지만, 그 결과로 로컬 디스크에 쓴 새 모델 파일은 컨테이너
   재시작·재배포 시 사라진다 — 영구 저장소(GCS 등)에 쓰도록 아키텍처를 먼저 바꿔야 하고,
   서빙 코드(`data_tools.py`)도 로컬 파일이 아니라 그 저장소에서 모델을 로드하도록 바꿔야
   한다. 이건 이번 작업 범위를 넘어서는 인프라 변경이다.

그래서 실제로 동작하지 않을 기능(트리거해도 아무 효과가 없거나 곧 사라지는 재학습)을 만드는
대신, **재학습·버전관리·게이트·드리프트체크 파이프라인 자체를 한 번에 실행 가능한 형태로
완성**해뒀다 — 원본 데이터가 있는 로컬/CI 환경에서 `python scripts/train_model.py` 한 줄로
전체 파이프라인이 돌아간다. 나중에 GCS 버킷을 모델 저장소로 추가하면, 이 파이프라인을 그대로
Cloud Scheduler → Cloud Run(또는 Cloud Build) 트리거에 연결할 수 있다.
