"""思考题 1：重现 L1 / L2 正则化的"系数路径"（coefficient path）。

数据来源：world 任务 layer 11 的训练集（500 条）中按方差选出的 8 个特征，
特征与目标都做了标准化，因此 λ 的取值对 L1/L2 可比。
L2 路径用 model.py 里手写的梯度下降求解，并与闭式解核对；L1 路径用 ISTA
（次梯度会在零点附近振荡，画不出精确为 0 的分段线性结构）。

输出：figures/coef_path_l1.png、figures/coef_path_l2.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import LinearRegressionModel
from train23 import load_split

HERE = Path(__file__).parent
FIG = HERE / "figures"

N_FEATURES_PICK = 8
LAMBDAS = np.logspace(-3, 1.5, 45)


def load_standardized(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = load_split("world", "train", 11)
    y = y[:, 0]  # 只用纬度作为演示目标
    # 按方差选特征
    var = x.var(axis=0)
    idx = np.argsort(var)[::-1][:n_features]
    xs = x[:, idx]
    ys = y

    xs = (xs - xs.mean(axis=0)) / xs.std(axis=0)
    ys = (ys - ys.mean()) / ys.std()
    print(f"选中的特征列（原始索引）: {idx.tolist()}")
    print(f"特征间最大|相关系数| = {np.max(np.abs(np.corrcoef(xs, rowvar=False) - np.eye(n_features))):.3f}")
    return xs, ys


def fit_path(xs: np.ndarray, ys: np.ndarray, lambda1: float, lambda2: float) -> np.ndarray:
    model = LinearRegressionModel(
        xs.shape[1],
        lambda1=lambda1,
        lambda2=lambda2,
        momentum=0.0,
        lr_decay=1e-4,
        max_iter=30_000,
        tol=1e-12,
        patience=200,
    )
    model.fit(xs, ys)
    return model.weights


def ista_path(xs: np.ndarray, ys: np.ndarray, lam1: float, iters: int = 5_000) -> np.ndarray:
    """L1 路径用近端梯度下降（ISTA）求解。

    model.py 按讲义用次梯度，固定步长会在零点附近振荡、无法精确得到 0；
    画路径时希望看到"精确为 0"的分段线性结构，因此这里对 L1 项用软阈值近端算子，
    平滑部分的梯度与 step 仍与讲义一致（步长取 1/L，L 为数据项的 Lipschitz 常数）。
    """
    n = xs.shape[0]
    # L = lambda_max((2/N) X^T X)
    v = np.ones(xs.shape[1]) / np.sqrt(xs.shape[1])
    for _ in range(50):
        u = xs @ v
        u /= max(np.linalg.norm(u), 1e-300)
        v = xs.T @ u
        v /= max(np.linalg.norm(v), 1e-300)
    lam_max = float(v @ (xs.T @ (xs @ v))) / n
    L = 2.0 * lam_max
    step = 1.0 / L
    w = np.zeros(xs.shape[1])
    for _ in range(iters):
        grad = (2.0 / n) * (xs.T @ (xs @ w - ys))
        z = w - step * grad
        w = np.sign(z) * np.maximum(np.abs(z) - step * lam1, 0.0)
    return w


def main() -> None:
    FIG.mkdir(exist_ok=True)
    xs, ys = load_standardized(N_FEATURES_PICK)

    paths_l1 = np.array([ista_path(xs, ys, lam) for lam in LAMBDAS])
    paths_l2 = np.array([fit_path(xs, ys, 0.0, lam) for lam in LAMBDAS])

    # 与 L2 闭式解核对，确认手写梯度下降收敛到最优
    g = xs.T @ xs / xs.shape[0]
    b = xs.T @ ys / xs.shape[0]
    closed = np.linalg.solve(g + LAMBDAS[-1] * np.eye(xs.shape[1]), b)
    print(f"L2 路径末点与闭式解最大偏差: {np.max(np.abs(paths_l2[-1] - closed)):.3e}")

    colors = plt.cm.tab10(np.linspace(0, 1, N_FEATURES_PICK))
    for name, paths in (("l1", paths_l1), ("l2", paths_l2)):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for j in range(N_FEATURES_PICK):
            ax.semilogx(LAMBDAS, paths[:, j], color=colors[j], lw=1.8, label=f"$w_{{{j+1}}}$")
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_xlabel(r"regularization strength $\lambda$")
        ax.set_ylabel(r"weight $w_j$")
        tex_name = r"$\ell_1$" if name == "l1" else r"$\ell_2$"
        ax.set_title(
            tex_name + " coefficient path (world layer11, 8 standardized features)"
        )
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        out = FIG / f"coef_path_{name}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"saved {out}")

    n_zero_l1 = (np.abs(paths_l1) < 1e-8).sum(axis=1)
    print("L1 路径上被压成 0 的系数个数（沿 λ 递增，每 4 个点采样）:", n_zero_l1[::4].tolist())
    print("L2 路径末端所有系数:", np.round(paths_l2[-1], 4).tolist())


if __name__ == "__main__":
    main()
