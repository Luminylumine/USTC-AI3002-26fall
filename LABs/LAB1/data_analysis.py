"""数据分布分析：训练集 vs 验证集（思考题 8 的证据，也用于思考题 7 的国家基线）。

覆盖两件事：
1. truth 任务：标签、句子长度、表征范数、均值漂移，以及"能否被线性分类器区分"
   （域分类器精度，越接近 0.5 说明分布越接近）；
2. world 任务：国家覆盖、坐标分布差异，以及按国家分组的均值基线误差。

输出：figures/dist_*.png，并在终端打印可直接写进报告的数字。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from train23 import load_split

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "datasets"


def truth_distribution() -> None:
    print("=" * 72)
    print("truth：训练集 vs 验证集")
    print("=" * 72)
    tr = pd.read_csv(DATA / "truth" / "train.csv")
    va = pd.read_csv(DATA / "truth" / "valid.csv")

    print(f"样本数: train={len(tr)}, valid={len(va)}")
    print(f"标签比例: train={tr['label'].mean():.4f}, valid={va['label'].mean():.4f}")

    for name, df in (("train", tr), ("valid", va)):
        length = df["sentence"].str.len()
        words = df["sentence"].str.split().str.len()
        print(f"{name:5s} 句长(字符) mean={length.mean():.2f} std={length.std():.2f} "
              f"median={length.median():.0f} | 词数 mean={words.mean():.2f}")

    ks_len = stats.ks_2samp(tr["sentence"].str.len(), va["sentence"].str.len())
    ks_word = stats.ks_2samp(tr["sentence"].str.split().str.len(), va["sentence"].str.split().str.len())
    print(f"句长 KS 检验: statistic={ks_len.statistic:.4f}, p={ks_len.pvalue:.3e}")
    print(f"词数 KS 检验: statistic={ks_word.statistic:.4f}, p={ks_word.pvalue:.3e}")

    # 重叠情况
    overlap = len(set(tr["sentence"]) & set(va["sentence"]))
    print(f"句子完全重复数: {overlap}")

    # 表征范数与均值漂移
    for layer in (11, 23, 35):
        xtr, _ = load_split("truth", "train", layer)
        xva, _ = load_split("truth", "valid", layer)
        ntr = np.linalg.norm(xtr, axis=1)
        nva = np.linalg.norm(xva, axis=1)
        ks = stats.ks_2samp(ntr, nva)
        shift = np.linalg.norm(xtr.mean(axis=0) - xva.mean(axis=0)) / np.linalg.norm(
            xtr.std(axis=0) + 1e-12
        )
        print(f"layer {layer:>2d}: 表征范数 train {ntr.mean():.2f}±{ntr.std():.2f} / "
              f"valid {nva.mean():.2f}±{nva.std():.2f} (KS p={ks.pvalue:.2e}); "
              f"均值漂移(按逐维std归一)={shift:.3f}")

    # 域分类器：能否用 layer23 特征区分 train/valid
    layer = 23
    xtr, _ = load_split("truth", "train", layer)
    xva, _ = load_split("truth", "valid", layer)
    x = np.vstack([xtr, xva])
    yd = np.concatenate([np.zeros(len(xtr)), np.ones(len(xva))])
    xa, xb, ya, yb = train_test_split(x, yd, test_size=0.4, random_state=0, stratify=yd)
    clf = LogisticRegression(max_iter=2000).fit(xa, ya)
    acc = clf.score(xb, yb)
    print(f"域分类器（区分 train/valid，layer{layer}）测试准确率 = {acc:.4f} "
          f"（0.5 表示不可区分，越高差异越大）")

    # 图：句长与表征范数分布
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].hist(tr["sentence"].str.len(), bins=40, alpha=0.6, label="train", density=True)
    axes[0].hist(va["sentence"].str.len(), bins=40, alpha=0.6, label="valid", density=True)
    axes[0].set_title("truth: sentence length")
    axes[0].set_xlabel("characters")
    axes[0].legend()
    xtr23, _ = load_split("truth", "train", 23)
    xva23, _ = load_split("truth", "valid", 23)
    axes[1].hist(np.linalg.norm(xtr23, axis=1), bins=40, alpha=0.6, label="train", density=True)
    axes[1].hist(np.linalg.norm(xva23, axis=1), bins=40, alpha=0.6, label="valid", density=True)
    axes[1].set_title("truth: representation norm (layer 23)")
    axes[1].set_xlabel("L2 norm")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "dist_truth.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def world_distribution() -> None:
    print("=" * 72)
    print("world：训练集 vs 验证集")
    print("=" * 72)
    tr = pd.read_csv(DATA / "world" / "train.csv")
    va = pd.read_csv(DATA / "world" / "valid.csv")

    tr_c, va_c = set(tr["country"]), set(va["country"])
    print(f"国家数: train={len(tr_c)}, valid={len(va_c)}, "
          f"验证集中未在训练集出现过的国家={len(va_c - tr_c)} "
          f"（涉及 {int(va['country'].isin(va_c - tr_c).sum())}/{len(va)} 条样本）")
    for name, df in (("train", tr), ("valid", va)):
        print(f"{name:5s} 纬度 mean={df.latitude.mean():.2f} std={df.latitude.std():.2f} "
              f"range=[{df.latitude.min():.1f},{df.latitude.max():.1f}] | "
              f"经度 mean={df.longitude.mean():.2f} std={df.longitude.std():.2f} "
              f"range=[{df.longitude.min():.1f},{df.longitude.max():.1f}]")
    print("纬度 KS:", f"{stats.ks_2samp(tr.latitude, va.latitude).statistic:.4f}")
    print("经度 KS:", f"{stats.ks_2samp(tr.longitude, va.longitude).statistic:.4f}")

    # 基线
    mean_const = tr[["latitude", "longitude"]].to_numpy().mean(axis=0)
    pred_const = np.tile(mean_const, (len(va), 1))
    true = va[["latitude", "longitude"]].to_numpy()
    mse_const = float(np.mean((pred_const - true) ** 2))

    g = tr.groupby("country")[["latitude", "longitude"]].mean()
    mlat = va["country"].map(g["latitude"])
    mlon = va["country"].map(g["longitude"])
    known = mlat.notna()
    pred_country = np.column_stack(
        [mlat.fillna(mean_const[0]), mlon.fillna(mean_const[1])]
    ).astype(float)
    mse_country = float(np.mean((pred_country - true) ** 2))
    mse_country_known = float(
        np.mean((pred_country[known.to_numpy()] - true[known.to_numpy()]) ** 2)
    )
    print(f"常数均值基线 validMSE = {mse_const:.2f}")
    print(f"国家均值基线 validMSE = {mse_country:.2f}（只看训练集见过的国家: {mse_country_known:.2f}）")
    print(f"同一国家内部的坐标方差（训练集）: lat {tr.groupby('country').latitude.var().mean():.2f}, "
          f"lon {tr.groupby('country').longitude.var().mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].scatter(tr.longitude, tr.latitude, s=8, alpha=0.5, label="train")
    axes[0].scatter(va.longitude, va.latitude, s=14, alpha=0.9, marker="x", label="valid")
    axes[0].set_title("world: coordinate distribution")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    counts = va["country"].value_counts()
    axes[1].bar(range(len(counts)), counts.values, color="tab:blue")
    axes[1].set_title(f"world: validation samples per country ({len(counts)} countries)")
    axes[1].set_xlabel("country (sorted by count)")
    axes[1].set_ylabel("#samples")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG / "dist_world.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    truth_distribution()
    world_distribution()


if __name__ == "__main__":
    main()
