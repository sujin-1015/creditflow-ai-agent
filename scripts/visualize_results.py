"""
모델링 결과 시각화 (matplotlib)

입력: ../processed/{val,test}.csv, ../models/{xgboost_model.joblib, metrics_report.json, threshold_config.json}
출력: ../models/figures/*.png

실행:
    python visualize_results.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = MODELS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
BLUE_MID = "#5598e7"
BLUE_LIGHT = "#9ec5f4"
GRAY = "#b7b9c2"
INK = "#14171c"
INK_SECOND = "#52586b"
MUTED = "#8a8d99"
GRID = "#e4e5ea"

plt.rcParams.update({
    "font.family": "Malgun Gothic",  # 한글 표시 (Windows 기본 고딕)
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECOND,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})

with open(MODELS_DIR / "metrics_report.json", encoding="utf-8") as f:
    metrics = json.load(f)


def fig1_model_comparison():
    fig, ax = plt.subplots(figsize=(6, 3.2))
    models = ["XGBoost\n(선정)", "LightGBM"]
    aucs = [metrics["val_auc"]["xgboost"], metrics["val_auc"]["lightgbm"]]
    colors = [BLUE, GRAY]

    bars = ax.barh(models, aucs, height=0.5, color=colors)
    ax.axvline(0.5, color=MUTED, linewidth=1)
    trans = ax.get_xaxis_transform()
    ax.text(0.5, 1.03, "random (0.5)", color=MUTED, fontsize=9, ha="center", transform=trans)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlabel("Validation AUC")
    ax.set_title("Baseline 모델 비교 (Val AUC)", fontsize=13, fontweight="bold", loc="left", pad=16)
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                 va="center", fontsize=11, fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "1_model_comparison.png", dpi=160)
    plt.close(fig)


def fig2_tier_bad_rate():
    tiers = ["승인", "조건부승인", "거절"]
    test_summary = metrics["test_tier_summary"]
    bad_rates = [test_summary[t]["bad_rate"] * 100 for t in ["approve", "conditional", "reject"]]
    shares = [test_summary[t]["share"] * 100 for t in ["approve", "conditional", "reject"]]
    counts = [test_summary[t]["count"] for t in ["approve", "conditional", "reject"]]
    overall_bad_rate = sum(
        metrics["test_confusion_matrix_reject_vs_actual"]["predicted_reject_as_positive"][k]
        for k in ["false_negative", "true_positive"]
    ) / sum(metrics["test_confusion_matrix_reject_vs_actual"]["predicted_reject_as_positive"].values()) * 100

    colors = [BLUE_LIGHT, BLUE_MID, BLUE_DARK]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.barh(tiers, bad_rates, height=0.55, color=colors)
    ax.axvline(overall_bad_rate, color=MUTED, linewidth=1)
    trans = ax.get_xaxis_transform()
    ax.text(overall_bad_rate, 1.03, f"전체 평균 {overall_bad_rate:.1f}%", color=MUTED, fontsize=9, ha="center", transform=trans)
    ax.set_xlim(0, max(bad_rates) * 1.3)
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlabel("실제 부도율 (%)")
    ax.set_title("판정 티어별 실제 부도율 (test set)", fontsize=13, fontweight="bold", loc="left", pad=16)
    for bar, rate, share, cnt in zip(bars, bad_rates, shares, counts):
        ax.text(rate + max(bad_rates) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{rate:.2f}%  ·  {cnt:,}건 ({share:.1f}%)",
                 va="center", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "2_tier_bad_rate.png", dpi=160)
    plt.close(fig)


def fig3_confusion_matrix():
    cm = metrics["test_confusion_matrix_reject_vs_actual"]["predicted_reject_as_positive"]
    tn, fp, fn, tp = cm["true_negative"], cm["false_positive"], cm["false_negative"], cm["true_positive"]
    matrix = np.array([[tn, fp], [fn, tp]])
    row_pct = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4.6))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            text_color = "white" if row_pct[i, j] > 0.55 else INK
            ax.text(j, i - 0.12, f"{labels[i][j]}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=text_color)
            ax.text(j, i + 0.10, f"{matrix[i, j]:,}", ha="center", va="center",
                     fontsize=15, fontweight="bold", color=text_color)
            ax.text(j, i + 0.32, f"(행 내 {row_pct[i, j]*100:.1f}%)", ha="center", va="center",
                     fontsize=9.5, color=text_color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["예측: 승인/조건부", "예측: 거절"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["실제: 정상 (0)", "실제: 부도 (1)"], fontsize=10)
    ax.set_title("Confusion Matrix — 거절 결정 기준 (test set)", fontsize=12.5, fontweight="bold", loc="left", pad=14)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tn + tp) / matrix.sum()
    fig.text(0.02, -0.02,
              f"Precision {precision*100:.1f}%   ·   Recall {recall*100:.1f}%   ·   Accuracy {accuracy*100:.1f}%",
              fontsize=10, color=INK_SECOND)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "3_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4_feature_importance():
    importance = pd.Series(metrics["feature_importance"]).sort_values(ascending=True)
    synth_features = {"revenue_volatility_synth", "debt_to_income_synth"}
    colors = [GRAY if name in synth_features else BLUE for name in importance.index]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    bars = ax.barh(importance.index, importance.values * 100, color=colors, height=0.65)
    ax.set_xlabel("Importance (%)")
    ax.set_title("Feature Importance — XGBoost (전체 18개 피처)", fontsize=13, fontweight="bold", loc="left")
    for bar, val in zip(bars, importance.values * 100):
        ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                 va="center", fontsize=9, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, importance.max() * 100 * 1.15)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=BLUE, label="실제 피처"),
        Patch(facecolor=GRAY, label="합성 노이즈 대조군 (실제 신호 없음, 검증용)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              fontsize=9.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "4_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    fig1_model_comparison()
    fig2_tier_bad_rate()
    fig3_confusion_matrix()
    fig4_feature_importance()
    print(f"저장 완료: {FIG_DIR}")


if __name__ == "__main__":
    main()
