"""
소상공인 대출 사전심사 에이전트 - 전처리 파이프라인

원본 데이터: ../Training Data.csv (읽기 전용, 절대 수정하지 않음)
출력 위치: ../processed/  (train.csv, val.csv, test.csv, encoding_maps.json, feature_manifest.md)

실행:
    python preprocess.py

주의:
- Training Data.csv에만 Risk_Flag(라벨)가 있으므로 이 파일만 사용해 자체적으로
  train/val/test를 분할한다 (Kaggle의 Test Data.csv는 라벨이 없어 학습/검증에 쓸 수 없음).
- revenue_volatility_synth, debt_to_income_synth 는 원본 데이터에 없는 값으로,
  실제 서비스라면 계좌/매출 API에서 가져와야 할 값을 PoC 데모를 위해 시드 고정
  난수로 합성한 것이다. 모델 성능 지표를 해석할 때 이 두 컬럼이 인위적으로
  타겟과 상관되게 만들어지지 않았다는 점(순수 노이즈 기반)을 감안해야 한다.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "Training Data.csv"
OUT_DIR = BASE_DIR / "processed"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
TARGET_ENCODE_SMOOTHING = 20  # 카테고리 표본이 적을수록 전체 평균 쪽으로 당겨짐

RENAME_MAP = {
    "Id": "id",
    "Income": "annual_revenue_krw",
    "Age": "age",
    "Experience": "career_years",
    "Married/Single": "household_status",
    "House_Ownership": "biz_premise_ownership",
    "Car_Ownership": "has_business_asset_car",
    "Profession": "industry_sector",
    "CITY": "biz_city",
    "STATE": "biz_region",
    "CURRENT_JOB_YRS": "biz_operation_years",
    "CURRENT_HOUSE_YRS": "biz_location_years",
    "Risk_Flag": "default_risk_label",
}

HIGH_CARDINALITY_COLS = ["industry_sector", "biz_city", "biz_region"]


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 위키백과 각주 잔재 제거: "Tiruchirappalli[10]" -> "Tiruchirappalli"
    df["CITY"] = df["CITY"].str.replace(r"\[\d+\]", "", regex=True).str.strip()
    # train/test 표기 통일 대비 (공백 <-> 언더스코어)
    for col in ["STATE", "Profession", "CITY"]:
        df[col] = df[col].str.replace(" ", "_").str.strip()
    return df


def rename_business_context(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(SEED)

    df["is_married"] = (df["household_status"] == "married").astype(int)
    df["has_car"] = (df["has_business_asset_car"] == "yes").astype(int)

    # 업력 안정성: 현 사업 운영 연차가 총 경력에서 차지하는 비중 (0으로 나누기 방지)
    df["job_stability_ratio"] = np.where(
        df["career_years"] > 0,
        df["biz_operation_years"] / df["career_years"],
        0.0,
    )

    df["income_per_age"] = df["annual_revenue_krw"] / df["age"]
    df["income_log"] = np.log1p(df["annual_revenue_krw"])

    # --- 합성(synthetic) 피처: 원본에 없는 매출 변동성/부채비율의 PoC 대체값 ---
    # 실제 서비스에서는 계좌/매출 API 연동으로 대체해야 함. 여기서는 재현 가능하도록
    # id를 시드로 사용한 결정론적 난수로 생성하고, 타겟과는 무관한 순수 노이즈다.
    row_seeds = df["id"].to_numpy()
    volatility = np.array([np.random.default_rng(SEED + s).uniform(0.05, 0.6) for s in row_seeds])
    debt_ratio = np.array([np.random.default_rng(SEED * 2 + s).uniform(0.1, 0.9) for s in row_seeds])
    df["revenue_volatility_synth"] = volatility
    df["debt_to_income_synth"] = debt_ratio

    return df


def split_data(df: pd.DataFrame, target: str = "default_risk_label"):
    """sklearn 없이 클래스별로 셔플 후 비율대로 잘라 stratified split을 재현한다."""
    rng = np.random.default_rng(SEED)
    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby(target):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))
        train_parts.append(group.loc[idx[:n_train]])
        val_parts.append(group.loc[idx[n_train:n_train + n_val]])
        test_parts.append(group.loc[idx[n_train + n_val:]])

    def _shuffle_concat(parts):
        combined = pd.concat(parts)
        shuffled_idx = rng.permutation(len(combined))
        return combined.iloc[shuffled_idx].reset_index(drop=True)

    return _shuffle_concat(train_parts), _shuffle_concat(val_parts), _shuffle_concat(test_parts)


def fit_target_encoding(train_df: pd.DataFrame, col: str, target: str = "default_risk_label"):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * TARGET_ENCODE_SMOOTHING) / (
        stats["count"] + TARGET_ENCODE_SMOOTHING
    )
    return smoothed.to_dict(), global_mean


def apply_target_encoding(df: pd.DataFrame, col: str, mapping: dict, global_mean: float) -> pd.Series:
    return df[col].map(mapping).fillna(global_mean)


def encode_features(train_df, val_df, test_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # 카디널리티 낮은 범주형: 원-핫 (train 기준 컬럼 집합 고정 후 val/test에 동일 적용)
    ownership_dummies_train = pd.get_dummies(train_df["biz_premise_ownership"], prefix="biz_premise_ownership")
    ownership_cols = ownership_dummies_train.columns.tolist()

    encoding_maps = {}
    for col in HIGH_CARDINALITY_COLS:
        mapping, global_mean = fit_target_encoding(train_df, col)
        encoding_maps[col] = {"mapping": mapping, "global_mean": global_mean}
        for df in (train_df, val_df, test_df):
            df[f"{col}_risk_te"] = apply_target_encoding(df, col, mapping, global_mean)

    out = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dummies = pd.get_dummies(df["biz_premise_ownership"], prefix="biz_premise_ownership")
        dummies = dummies.reindex(columns=ownership_cols, fill_value=0)
        df = pd.concat([df, dummies], axis=1)
        out[name] = df

    return out["train"], out["val"], out["test"], encoding_maps


FEATURE_COLUMNS = [
    "annual_revenue_krw",
    "income_log",
    "income_per_age",
    "age",
    "career_years",
    "biz_operation_years",
    "biz_location_years",
    "job_stability_ratio",
    "is_married",
    "has_car",
    "revenue_volatility_synth",
    "debt_to_income_synth",
    "industry_sector_risk_te",
    "biz_city_risk_te",
    "biz_region_risk_te",
    "biz_premise_ownership_owned",
    "biz_premise_ownership_rented",
    "biz_premise_ownership_norent_noown",
]
LABEL_COLUMN = "default_risk_label"
ID_COLUMN = "id"


def save_outputs(train_df, val_df, test_df, encoding_maps):
    OUT_DIR.mkdir(exist_ok=True)

    keep_cols = [ID_COLUMN] + FEATURE_COLUMNS + [LABEL_COLUMN]
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} 데이터프레임에 예상 컬럼이 없습니다: {missing}")
        df[keep_cols].to_csv(OUT_DIR / f"{name}.csv", index=False)

    with open(OUT_DIR / "encoding_maps.json", "w", encoding="utf-8") as f:
        json.dump(encoding_maps, f, ensure_ascii=False, indent=2, default=float)

    manifest = f"""# 전처리 산출물 매니페스트

원본 파일(`Training Data.csv`)은 수정하지 않았으며, 아래 파일은 모두 `processed/` 폴더에 새로 생성되었다.

## 분할
- train.csv : {len(train_df)}행 ({SPLIT_RATIOS['train']*100:.0f}%)
- val.csv   : {len(val_df)}행 ({SPLIT_RATIOS['val']*100:.0f}%)
- test.csv  : {len(test_df)}행 ({SPLIT_RATIOS['test']*100:.0f}%)
- 분할은 `default_risk_label` 기준 stratified split (seed={SEED})

## 클래스 비율 (default_risk_label=1 비율)
- train: {train_df[LABEL_COLUMN].mean():.4f}
- val:   {val_df[LABEL_COLUMN].mean():.4f}
- test:  {test_df[LABEL_COLUMN].mean():.4f}

## 피처 컬럼 ({len(FEATURE_COLUMNS)}개)
{chr(10).join(f'- `{c}`' for c in FEATURE_COLUMNS)}

## 주의
- `industry_sector_risk_te`, `biz_city_risk_te`, `biz_region_risk_te` 는 train에서만 학습한
  target(mean) encoding이며, 스무딩 파라미터는 {TARGET_ENCODE_SMOOTHING}. val/test에는 train에서
  학습한 매핑을 그대로 적용했고(리키지 방지), train에 없던 카테고리는 global mean으로 대체했다.
  매핑 값은 `encoding_maps.json`에 저장.
- `revenue_volatility_synth`, `debt_to_income_synth` 는 원본에 없는 값으로, id 기반 시드 고정
  난수로 합성한 PoC placeholder다 (타겟과 상관 없는 순수 노이즈). 실 서비스 연동 전까지는
  모델 성능 해석 시 이 두 피처의 기여도를 과신하지 않아야 한다.
"""
    with open(OUT_DIR / "feature_manifest.md", "w", encoding="utf-8") as f:
        f.write(manifest)


def main():
    raw = load_raw()
    df = clean_categoricals(raw)
    df = rename_business_context(df)
    df = engineer_features(df)

    train_df, val_df, test_df = split_data(df)
    train_df, val_df, test_df, encoding_maps = encode_features(train_df, val_df, test_df)

    save_outputs(train_df, val_df, test_df, encoding_maps)

    print("전처리 완료")
    print(f"  원본: {RAW_PATH} (읽기 전용, 변경 없음)")
    print(f"  출력: {OUT_DIR}")
    print(f"  train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"  피처 수: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
