"""思考题 2：用一个可控案例说明"用 L2 做稀疏特征选择"为什么不好（相比 L1）。

构造：18 维特征，其中 x1,x2,x3 是真信号（系数 3, 2, -1），x6、x7 分别是
x1、x2 的强相关副本（现实里常见：两个高度相关的特征都携带同一信息），
其余为纯噪声；目标 = 线性组合 + 观测噪声。

L1 用 ISTA（近端梯度，次梯度法在零点附近振荡、得不到精确 0），L2 用 model.py
的梯度下降求解；数据确定性切成训练/验证/测试三份，λ 与"稀疏口径"都只在
验证集上选，测试集全程不参与选择，最后报告一次最终测试 MSE，避免用测试集
调参造成的选择泄漏。
比较：非零系数个数、是否能恢复真实支撑集、相关特征上的权重分配、测试 MSE。

输出：figures/feature_selection_l1_l2.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coef_path import ista_path
from model import LinearRegressionModel

HERE = Path(__file__).parent
FIG = HERE / "figures"

rng = np.random.default_rng(0)
N_POOL = 900
N_TRAIN, N_VALID, N_TEST = 300, 300, 300
D = 18
TRUE_W = np.zeros(D)
TRUE_W[:3] = [3.0, 2.0, -1.0]
TRUE_SUPPORT = [0, 1, 2]
LAMBDAS = np.logspace(-3, 0.5, 30)


def make_pool(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, D))
    # 相关副本：x6 = 0.95*x1 + 噪声，x7 = 0.95*x2 + 噪声
    x[:, 5] = 0.95 * x[:, 0] + rng.normal(scale=0.3, size=n)
    x[:, 6] = 0.95 * x[:, 1] + rng.normal(scale=0.3, size=n)
    y = x @ TRUE_W + rng.normal(scale=0.5, size=n)
    return x, y


def fit(x: np.ndarray, y: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    # L1 用次梯度，带动量会在零点附近发散，按讲义用固定步长（配合 lr 衰减）
    m = LinearRegressionModel(
        D,
        lambda1=lam1,
        lambda2=lam2,
        momentum=0.0,
        lr_decay=1e-4,
        max_iter=60_000,
        tol=1e-12,
        patience=500,
    )
    m.fit(x, y)
    return m.weights


def main() -> None:
    FIG.mkdir(exist_ok=True)
    # 确定性切分：先生成一个固定池再按连续区间切成 train/valid/test，
    # 只依赖固定种子 rng，重复运行结果完全一致。
    x_all, y_all = make_pool(N_POOL)
    xtr, ytr = x_all[:N_TRAIN], y_all[:N_TRAIN]
    xva, yva = x_all[N_TRAIN:N_TRAIN + N_VALID], y_all[N_TRAIN:N_TRAIN + N_VALID]
    xte, yte = x_all[N_TRAIN + N_VALID:], y_all[N_TRAIN + N_VALID:]
    assert len(xte) == N_TEST
    print(f"确定性切分：train={N_TRAIN} valid={N_VALID} test={N_TEST}（test 不参与选择）")

    best, sparse, records_all = {}, {}, {}
    for tag, use_l1 in (("l1", True), ("l2", False)):
        records = []
        for lam in LAMBDAS:
            if use_l1:
                w = ista_path(xtr, ytr, lam, iters=20_000)
            else:
                w = fit(xtr, ytr, 0.0, lam)
            mse = float(np.mean((xva @ w - yva) ** 2))
            records.append((mse, lam, w))
        records_all[tag] = records
        mse, lam, w = min(records, key=lambda r: r[0])
        best[tag] = dict(lam=lam, w=w, valid_mse=mse)
        # 稀疏口径：验证 MSE 不超过最优 5% 的 λ 中取最大者（更偏向稀疏解）
        cand = [r for r in records if r[0] <= 1.05 * best[tag]["valid_mse"]]
        mse, lam, w = max(cand, key=lambda r: r[1])
        sparse[tag] = dict(lam=lam, w=w, valid_mse=mse)

    # 只在选完 λ 之后，才在测试集上评估一次（无选择泄漏）
    for tag in ("l1", "l2"):
        best[tag]["test_mse"] = float(np.mean((xte @ best[tag]["w"] - yte) ** 2))
        sparse[tag]["test_mse"] = float(np.mean((xte @ sparse[tag]["w"] - yte) ** 2))

    print("真实系数      :", TRUE_W[:8].tolist(), "...")
    for tag in ("l1", "l2"):
        print(
            f"{tag.upper()} 最优 lam={best[tag]['lam']:.4g} 验证MSE={best[tag]['valid_mse']:.4f}"
            f" 测试MSE={best[tag]['test_mse']:.4f}"
            f"   |  稀疏口径 lam={sparse[tag]['lam']:.4g}"
            f" 验证MSE={sparse[tag]['valid_mse']:.4f} 测试MSE={sparse[tag]['test_mse']:.4f}"
        )
        print(f"  {tag.upper()} 系数前 8 :", np.round(best[tag]['w'][:8], 4).tolist())
        print(f"  {tag.upper()} 稀疏口径前 8 :", np.round(sparse[tag]['w'][:8], 4).tolist())

    print("--- 用稀疏口径的 λ（验证 MSE 仍在最优 5% 内）比较 ---")
    for tag in ("l1", "l2"):
        w = sparse[tag]["w"]
        nz = int(np.sum(np.abs(w) > 1e-8))
        support = sorted(np.where(np.abs(w) > 1e-8)[0].tolist())
        noise_max = float(np.max(np.abs(w[3:])))
        print(
            f"{tag.upper()}: 非零系数个数={nz}（真实支撑集大小=3） 支撑集={support}"
            f" 精确恢复={support == TRUE_SUPPORT} 噪声特征最大|权重|={noise_max:.4f}"
        )
    COPY_IDX = [5, 6]  # x6、x7 是 x1、x2 的相关副本
    pure_noise = [i for i in range(3, D) if i not in COPY_IDX]
    w1s, w2s = sparse["l1"]["w"], sparse["l2"]["w"]
    print(
        f"相关副本上的权重：L1  x1={w1s[0]:.3f}, x6={w1s[5]:.3f}"
        f"   L2  x1={w2s[0]:.3f}, x6={w2s[5]:.3f}"
    )
    for tag, w in (("L1", w1s), ("L2", w2s)):
        print(
            f"{tag} 稀疏口径下：纯噪声特征最大|权重|={np.max(np.abs(w[pure_noise])):.4f}"
            f"  相关副本最大|权重|={np.max(np.abs(w[COPY_IDX])):.4f}"
        )
    print(
        "结论：同一 λ 口径下 L1 把纯噪声精确置 0（只需再决定相关副本去留），"
        "L2 永远不为 0、想剔除噪声必须人为设阈值"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, tag, title in (
        (axes[0], "l1", rf"$\ell_1$ (sparse $\lambda$={sparse['l1']['lam']:.3g})"),
        (axes[1], "l2", rf"$\ell_2$ (sparse $\lambda$={sparse['l2']['lam']:.3g})"),
    ):
        w = sparse[tag]["w"]
        cols = ["tab:green" if i in TRUE_SUPPORT else "tab:gray" for i in range(D)]
        ax.bar(np.arange(1, D + 1), w, color=cols)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(title)
        ax.set_xlabel("feature index")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("weight")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    fig.suptitle("green = true signal features (x1,x2,x3); gray = noise / correlated copies")
    fig.tight_layout()
    out = FIG / "feature_selection_l1_l2.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
