"""思考题 6/7：验证集预测可视化与"国家均值基线"对照。

说明：为避免地图边界带来的合规风险，这里只画经纬度散点（不绘制任何国界线），
坐标轴等比例，横轴经度、纵轴纬度。

输出：
- figures/valid_predictions.png   真实坐标 vs 预测坐标（按误差着色）
- figures/baseline_compare.png    我们的模型 / 国家均值基线 / 全局均值基线 的验证 MSE
- figures/residual_vs_country.png 扣除国家均值后的残差对照（回答思考题 7，
  仅在训练集中出现过的国家的验证样本上评估）
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train23 import load_split

HERE = Path(__file__).parent
FIG = HERE / "figures"
RESULTS = HERE / "results"


def main() -> None:
    FIG.mkdir(exist_ok=True)
    choice = json.loads((RESULTS / "chosen.json").read_text(encoding="utf-8"))
    layer = choice["world"]["layer"]

    with np.load(RESULTS / "world.npz", allow_pickle=False) as data:
        weights = data["weights"]
        bias = data["bias"]
    print(f"使用 layer={layer} 的提交权重，weights.shape={weights.shape}, bias={bias}")

    xva, yva = load_split("world", "valid", layer)
    pred = np.column_stack([xva @ weights[0] + bias[0], xva @ weights[1] + bias[1]])
    true = yva
    err = np.linalg.norm(pred - true, axis=1)
    mse = float(np.mean((pred - true) ** 2))
    mse_lat = float(np.mean((pred[:, 0] - true[:, 0]) ** 2))
    mse_lon = float(np.mean((pred[:, 1] - true[:, 1]) ** 2))
    print(f"我们的模型 validation 平均 MSE（纬度、经度两维平均）= {mse:.4f}"
          f"（纬度 MSE={mse_lat:.4f}, 经度 MSE={mse_lon:.4f}；"
          f"纬+经之和 {mse_lat + mse_lon:.4f} 是“纬度 MSE + 经度 MSE”，不是 MSE），"
          f"平均大圆距离近似 = {np.mean(err)*111:.1f} km")

    tr_csv = pd.read_csv(HERE / "datasets" / "world" / "train.csv")
    va_csv = pd.read_csv(HERE / "datasets" / "world" / "valid.csv")
    mean_const = tr_csv[["latitude", "longitude"]].to_numpy().mean(axis=0)
    g = tr_csv.groupby("country")[["latitude", "longitude"]].mean()
    mlat = va_csv["country"].map(g["latitude"])
    mlon = va_csv["country"].map(g["longitude"])
    known = mlat.notna().to_numpy()
    pred_country = np.column_stack(
        [mlat.fillna(mean_const[0]), mlon.fillna(mean_const[1])]
    ).astype(float)
    pred_const = np.tile(mean_const, (len(true), 1))

    mse_ours = float(np.mean((pred - true) ** 2))
    mse_country = float(np.mean((pred_country - true) ** 2))
    mse_const = float(np.mean((pred_const - true) ** 2))
    mse_ours_known = float(np.mean((pred[known] - true[known]) ** 2))
    mse_country_known = float(np.mean((pred_country[known] - true[known]) ** 2))
    print(f"MSE  我们={mse_ours:.2f} | 国家均值={mse_country:.2f} | 全局均值={mse_const:.2f}")
    print(f"仅训练集见过的国家({known.sum()} 条): 我们={mse_ours_known:.2f}, "
          f"国家均值={mse_country_known:.2f}")

    # ---- 图 1：真实 vs 预测散点（无国界线） ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sc0 = axes[0].scatter(true[:, 1], true[:, 0], c=err, cmap="viridis", s=26)
    axes[0].set_title("ground truth coordinates (colored by our error)")
    sc1 = axes[1].scatter(pred[:, 1], pred[:, 0], c=err, cmap="viridis", s=26)
    axes[1].set_title("model predictions (colored by our error)")
    for ax, sc in zip(axes, (sc0, sc1)):
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
    fig.colorbar(sc0, ax=axes, label="error (degrees)", shrink=0.85)
    fig.suptitle(f"world task, layer {layer}: validation predictions (no borders drawn)")
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.12, top=0.86, wspace=0.22)
    out1 = FIG / "valid_predictions.png"
    fig.savefig(out1, dpi=160)
    plt.close(fig)
    print(f"saved {out1}")

    # ---- 图 2：三种模型的 MSE 对比 ----
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    names = ["our model\n(linear probe)", "country-mean\nbaseline", "global-mean\nbaseline"]
    values = [mse_ours, mse_country, mse_const]
    bars = ax.bar(names, values, color=["tab:blue", "tab:orange", "tab:gray"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
    ax.set_ylabel("mean validation MSE over lat/lon (degrees$^2$)")
    ax.set_title("validation MSE (per-coordinate mean): our model vs baselines")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out2 = FIG / "baseline_compare.png"
    fig.savefig(out2, dpi=160)
    plt.close(fig)
    print(f"saved {out2}")

    # ---- 图 3：扣除国家均值后的残差（回答思考题 7） ----
    # 严格只在“训练集中出现过的国家”的验证样本上评估：
    # 这样“国家均值”对每条样本都有明确定义，残差分析不被未见国家的
    # 全局均值退化所混入。相关系数与残差 MSE 都在该子集上计算。
    true_k = true[known]
    pred_k = pred[known]
    country_mean_k = pred_country[known]
    resid_true = true_k - country_mean_k
    resid_pred = pred_k - country_mean_k
    r = float(np.corrcoef(resid_true[:, 0], resid_pred[:, 0])[0, 1])
    r_lon = float(np.corrcoef(resid_true[:, 1], resid_pred[:, 1])[0, 1])
    # 模型残差预测误差 vs “残差恒为 0”的基线（即完全不预测国家内部偏移）
    within_mse_ours = float(np.mean((resid_pred - resid_true) ** 2))
    within_mse_zero = float(np.mean(resid_true**2))
    print(f"仅训练集见过的国家（{int(known.sum())} 条）残差分析：")
    print(f"  纬度残差相关={r:.4f}，经度残差相关={r_lon:.4f}")
    print(f"  残差预测误差 MSE（我们的残差 - 真实残差）= {within_mse_ours:.2f}，"
          f"零残差基线 MSE（真实残差^2）= {within_mse_zero:.2f}，"
          f"改善={within_mse_zero - within_mse_ours:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for k, (ax, name, rr) in enumerate(zip(axes, ("latitude", "longitude"), (r, r_lon))):
        ax.scatter(resid_true[:, k], resid_pred[:, k], s=22, alpha=0.75)
        lim = np.abs(np.concatenate([resid_true[:, k], resid_pred[:, k]])).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax.set_xlabel(f"true - country mean ({name})")
        ax.set_ylabel(f"predicted - country mean ({name})")
        ax.set_title(f"within-country residual, corr={rr:.3f}")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"known-country validation samples only (n={int(known.sum())}), "
        f"residual MSE={within_mse_ours:.1f} vs zero-residual {within_mse_zero:.1f}"
    )
    fig.tight_layout()
    out3 = FIG / "residual_vs_country.png"
    fig.savefig(out3, dpi=160)
    plt.close(fig)
    print(f"saved {out3}")

    # 未见过国家上的表现
    unseen = ~known
    if unseen.any():
        print(f"验证集中训练未见国家的样本 {int(unseen.sum())} 条: "
              f"我们的 MSE={float(np.mean((pred[unseen]-true[unseen])**2)):.2f}, "
              f"国家均值基线(退化为全局均值) MSE="
              f"{float(np.mean((pred_country[unseen]-true[unseen])**2)):.2f}")


if __name__ == "__main__":
    main()
