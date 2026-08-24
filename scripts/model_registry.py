"""모델 아티팩트 버전 관리 + 성능 게이트.

지금까지 train_model.py는 models/xgboost_model.joblib을 실행할 때마다 덮어썼다 —
이전 버전이 뭐였는지, 언제 학습됐는지, 성능이 실제로 개선됐는지 알 방법이 없었다.

이 모듈은:
1. 학습마다 models/versions/{version_id}/ 에 아티팩트 스냅샷을 남기고
2. models/model_registry.json에 버전 이력(타임스탬프·AUC·git 커밋·승격 여부)을 기록하고
3. 새 후보 모델의 test AUC가 현재 프로덕션(마지막으로 승격된 버전)보다 낮으면
   **models/xgboost_model.joblib(실서빙 경로)을 덮어쓰지 않고** 알림만 남긴다(성능 게이트).

models/versions/의 바이너리 스냅샷은 .gitignore로 제외했다 — 저장소에 남기는 건 작고
사람이 읽을 수 있는 model_registry.json(이력)이지, 버전마다 쌓이는 수백 KB 바이너리가 아니다.
"""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
VERSIONS_DIR = MODELS_DIR / "versions"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# 프로덕션 서빙 경로 — data_tools.py가 그대로 읽는 고정 경로라 바꾸지 않는다.
LIVE_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
LIVE_LGBM_PATH = MODELS_DIR / "lightgbm_model.joblib"
LIVE_THRESHOLD_PATH = MODELS_DIR / "threshold_config.json"


def _git_short_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 — 버전 기록에 git 정보는 있으면 좋고 없어도 무방
        return None


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"versions": []}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def current_production_entry() -> Optional[dict]:
    """가장 최근에 승격(promoted=True)된 버전을 반환한다. 아직 하나도 없으면 None."""
    registry = _load_registry()
    promoted = [v for v in registry["versions"] if v.get("promoted")]
    return promoted[-1] if promoted else None


def evaluate_gate(candidate_test_auc: float, min_improvement: float = 0.0) -> dict:
    """새 후보의 test_auc를 현재 프로덕션과 비교해 승격 여부를 결정한다.

    첫 버전(레지스트리가 비어있음)은 비교 대상이 없으므로 무조건 승격한다.
    min_improvement는 허용 오차(예: 0.001을 주면 아주 미세한 하락은 눈감아줌) — 기본값 0.0은
    "조금이라도 낮으면 거부"라는 사용자 요구사항을 엄격하게 따른다.
    """
    current = current_production_entry()
    if current is None:
        return {"promote": True, "reason": "첫 등록 버전(비교 대상 없음) — 자동 승격", "baseline_test_auc": None}

    baseline_auc = current["test_auc"]
    if candidate_test_auc >= baseline_auc - min_improvement:
        return {
            "promote": True,
            "reason": f"새 test_auc({candidate_test_auc:.4f})가 현재 프로덕션({baseline_auc:.4f}) 이상",
            "baseline_test_auc": baseline_auc,
        }
    return {
        "promote": False,
        "reason": (
            f"새 test_auc({candidate_test_auc:.4f})가 현재 프로덕션({baseline_auc:.4f})보다 낮음 "
            f"— 성능 게이트 차단, 배포하지 않음"
        ),
        "baseline_test_auc": baseline_auc,
    }


def save_version(
    metrics: dict,
    threshold_result: dict,
    xgb_model_path: Path,
    lgbm_model_path: Path,
    promote: bool,
    gate_result: dict,
) -> dict:
    """방금 학습된 후보 아티팩트를 versions/{version_id}/에 스냅샷하고 레지스트리에 기록한다.
    promote=True면 실서빙 경로(models/xgboost_model.joblib 등)도 이 버전으로 갱신한다."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_hash = _git_short_hash()
    version_id = f"{timestamp}_{git_hash}" if git_hash else timestamp

    version_dir = VERSIONS_DIR / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xgb_model_path, version_dir / "xgboost_model.joblib")
    shutil.copy2(lgbm_model_path, version_dir / "lightgbm_model.joblib")
    (version_dir / "threshold_config.json").write_text(
        json.dumps(threshold_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (version_dir / "metrics_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    entry = {
        "version_id": version_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "primary_model": metrics["primary_model"],
        "val_auc": metrics["val_auc"],
        "test_auc": metrics["test_auc"],
        "thresholds": {"t_approve": threshold_result["t_approve"], "t_reject": threshold_result["t_reject"]},
        "promoted": promote,
        "gate_result": gate_result,
    }

    registry = _load_registry()
    if promote:
        # 이전에 승격됐던 버전들의 promoted 플래그는 이력으로만 남기고 해제한다
        # (current_production_entry()가 "가장 최근 promoted=True"만 보게 하기 위함).
        for v in registry["versions"]:
            v["promoted"] = False
    registry["versions"].append(entry)
    _save_registry(registry)

    if promote:
        shutil.copy2(xgb_model_path, LIVE_MODEL_PATH)
        shutil.copy2(lgbm_model_path, LIVE_LGBM_PATH)
        LIVE_THRESHOLD_PATH.write_text(json.dumps(threshold_result, ensure_ascii=False, indent=2), encoding="utf-8")

    return entry


def list_versions() -> list[dict]:
    return _load_registry()["versions"]


if __name__ == "__main__":
    current = current_production_entry()
    if current is None:
        print("등록된 버전이 없습니다.")
    else:
        print(f"현재 프로덕션 버전: {current['version_id']} (test_auc={current['test_auc']:.4f})")
    print(f"\n전체 버전 이력 ({len(list_versions())}개):")
    for v in list_versions():
        tag = "PRODUCTION" if v["promoted"] else "rejected" if not v["gate_result"].get("promote", True) else "superseded"
        print(f"  {v['version_id']} | test_auc={v['test_auc']:.4f} | {tag}")
