# 모델 학습 및 판정 임계값 리포트

**버전**: 20260824T175139Z_4507445 (git 4507445) — 이력은 `models/model_registry.json` 참고

## 1. Baseline 모델 비교 (val set AUC)
| 모델 | Val AUC |
|---|---|
| XGBoost | 0.7846 |
| LightGBM | 0.7794 |

**Primary 모델**: xgboost (val AUC 기준 채택)

## 2. 판정 임계값 (val set에서 탐색, 완화 단계: 0)
- 승인 임계값 (t_approve): 0.5877 → 확률 < t_approve 인 경우 승인
- 거절 임계값 (t_reject): 0.6264 → 확률 >= t_reject 인 경우 거절
- 그 사이는 조건부승인

### Val set 티어별 분포
| tier        |   count |   bad_rate |   share |
|:------------|--------:|-----------:|--------:|
| approve     |   32130 |  0.0786181 |    0.85 |
| conditional |    1890 |  0.250265  |    0.05 |
| reject      |    3780 |  0.436508  |    0.1  |

## 3. Test set(held-out) 최종 성능
- **Test AUC**: 0.7883

### Test set 티어별 분포
| tier        |   count |   bad_rate |     share |
|:------------|--------:|-----------:|----------:|
| approve     |   32110 |  0.0765805 | 0.849471  |
| conditional |    1883 |  0.280935  | 0.0498148 |
| reject      |    3807 |  0.436564  | 0.100714  |

### Confusion Matrix (거절 결정 = positive, 실제 부도 = positive 기준)
|  | 실제 정상(0) | 실제 부도(1) |
|---|---|---|
| **승인/조건부 (미거절)** | TN=31005 | FN=2988 |
| **거절** | FP=2145 | TP=1662 |

- Precision(거절 결정이 실제 부도일 확률): 0.4366
- Recall(실제 부도 중 거절로 잡아낸 비율): 0.3574

## 4. 참고: Feature Importance 상위 항목 (XGBoost)
|                                    |   importance |
|:-----------------------------------|-------------:|
| biz_city_risk_te                   |    0.135494  |
| has_car                            |    0.0690291 |
| biz_premise_ownership_owned        |    0.0643266 |
| biz_premise_ownership_norent_noown |    0.0616834 |
| industry_sector_risk_te            |    0.0612164 |
| biz_operation_years                |    0.0572007 |
| career_years                       |    0.0570671 |
| age                                |    0.0566517 |
| is_married                         |    0.0548954 |
| income_per_age                     |    0.0540167 |
| biz_region_risk_te                 |    0.0526442 |
| job_stability_ratio                |    0.052195  |
| biz_premise_ownership_rented       |    0.0520813 |
| biz_location_years                 |    0.0510856 |
| annual_revenue_krw                 |    0.0493537 |
| income_log                         |    0.0472072 |
| revenue_volatility_synth           |    0.0124794 |
| debt_to_income_synth               |    0.011373  |

## 주의
- EDA 단계에서 Income/Age/Experience 등 개별 수치 피처와 타겟의 단순 상관계수는 거의 0에
  가까웠지만, `biz_city`/`industry_sector`/`biz_region`의 그룹별 실제 부도율 차이(예: 지역별
  4.6%~21.6%, 표본 수천~수만 건)는 통계적으로 유의미한 수준이었다. 이 범주형 target encoding이
  AUC 상승의 주된 기여 요인이며(feature importance 참고), 개별 수치형 피처는 상호작용을 통해
  보조적으로 기여한다.
- `revenue_volatility_synth`, `debt_to_income_synth`는 원본에 없는 순수 합성 노이즈로,
  importance가 가장 낮게 나온 것으로 실제 신호가 아님을 확인했다 (검증 목적으로 의도적으로 포함).
- val AUC(0.7846)와 test AUC(0.7883)가 근접해 특정 split에 대한 우연한
  과적합이 아니라 일반화되는 패턴임을 확인했다.
