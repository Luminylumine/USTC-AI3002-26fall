"""思考题 3：与 sklearn 实现的拟合效果对比（同一数据、同一正则化语义）。

正则化强度的换算：
- 线性：我们最小化  mean((Xw-y)^2) + lambda2*||w||^2 ；
  sklearn Ridge 最小化 ||Xw-y||^2 + alpha*||w||^2  →  alpha = N * lambda2
- 逻辑：我们最小化  mean(NLL) + lambda2*||w||^2 ；
  sklearn LogisticRegression 最小化 sum(NLL) + 1/(2C)*||w||^2
  →  lambda2 = 1/(2*C*N)  →  C = 1/(2*N*lambda2)

输出：打印两者的训练/验证指标、权重差异与耗时。
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import Ridge as SKRidge
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from model import LinearRegressionModel, LogisticRegressionModel
from train23 import load_split


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def tiny_example() -> None:
    section("3.1 小例子（train1.py 的数据：8 条、2 维特征）")
    features = np.array(
        [[-2, -1], [-1, -1], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 1], [2, 1]],
        dtype=np.float64,
    )
    y = 2 * features[:, 0] - features[:, 1] + 1

    ours = LinearRegressionModel(2, max_iter=50_000, tol=1e-14, patience=500, momentum=0.99)
    ours.fit(features, y)
    sk = SKLinearRegression()
    sk.fit(features, y)
    print(f"ours  w={np.round(ours.weights,6).tolist()} b={ours.bias:.6f} "
          f"MSE={np.mean((ours.predict(features)-y)**2):.3e} iters={ours.n_iter_}")
    print(f"sklearn w={np.round(sk.coef_,6).tolist()} b={sk.intercept_:.6f} "
          f"MSE={np.mean((sk.predict(features)-y)**2):.3e}")
    print(f"权重最大差异 = {np.max(np.abs(ours.weights - sk.coef_)):.3e}")


def world_ridge(layer: int = 11, lam2_lat: float = 0.14, lam2_lon: float = 0.28) -> None:
    section(f"3.2 world（layer {layer}）线性回归：手写 GD vs sklearn Ridge（用最终选出的 λ）")
    xtr, ytr = load_split("world", "train", layer)
    xva, yva = load_split("world", "valid", layer)
    n = xtr.shape[0]

    print("[a] 原始特征（与实验里 sweep 用的输入一致）")
    for out, (name, lam2) in enumerate((("latitude", lam2_lat), ("longitude", lam2_lon))):
        alpha = n * lam2
        ours = LinearRegressionModel(
            xtr.shape[1], lambda2=lam2, momentum=0.99, max_iter=20_000, tol=1e-10, patience=100
        )
        t0 = time.time()
        ours.fit(xtr, ytr[:, out])
        t_ours = time.time() - t0

        sk = SKRidge(alpha=alpha)
        t0 = time.time()
        sk.fit(xtr, ytr[:, out])
        t_sk = time.time() - t0

        mse_ours = np.mean((ours.predict(xva) - yva[:, out]) ** 2)
        mse_sk = np.mean((sk.predict(xva) - yva[:, out]) ** 2)
        cos = float(
            ours.weights @ sk.coef_ / (np.linalg.norm(ours.weights) * np.linalg.norm(sk.coef_))
        )
        # 目标函数值（同一正则化语义下的可比量）
        r_o = xtr @ ours.weights + ours.bias - ytr[:, out]
        r_s = xtr @ sk.coef_ + float(sk.intercept_) - ytr[:, out]
        j_o = float(np.mean(r_o**2) + lam2 * ours.weights @ ours.weights)
        j_s = float(np.mean(r_s**2) + lam2 * sk.coef_ @ sk.coef_)
        print(f"  {name:9s} ours validMSE={mse_ours:9.4f} ({t_ours:.1f}s, {ours.n_iter_} iters) | "
              f"sklearn validMSE={mse_sk:9.4f} ({t_sk:.1f}s) | cos(w)={cos:.6f} | "
              f"max|dw|={np.max(np.abs(ours.weights - sk.coef_)):.3e}")
        print(f"           训练目标 J=mean(r^2)+λ2||w||^2：ours={j_o:.6f}  sklearn={j_s:.6f}"
              f"（差 {j_o - j_s:+.2e}，手写 GD 在病态方向上未完全收敛）")

    print("[b] 特征 z-score 标准化后（两边输入完全相同，验证同一正则化语义）")
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    xs, xvs = (xtr - mu) / sd, (xva - mu) / sd
    for out, (name, lam2) in enumerate((("latitude", lam2_lat), ("longitude", lam2_lon))):
        ours = LinearRegressionModel(
            xs.shape[1], lambda2=lam2, momentum=0.99, max_iter=20_000, tol=1e-12, patience=200
        )
        t0 = time.time()
        ours.fit(xs, ytr[:, out])
        t_ours = time.time() - t0
        sk = SKRidge(alpha=n * lam2)
        t0 = time.time()
        sk.fit(xs, ytr[:, out])
        t_sk = time.time() - t0
        mse_ours = np.mean((ours.predict(xvs) - yva[:, out]) ** 2)
        mse_sk = np.mean((sk.predict(xvs) - yva[:, out]) ** 2)
        print(f"  {name:9s} ours validMSE={mse_ours:9.4f} ({t_ours:.1f}s, {ours.n_iter_} iters) | "
              f"sklearn validMSE={mse_sk:9.4f} ({t_sk:.1f}s) | "
              f"max|dw|={np.max(np.abs(ours.weights - sk.coef_)):.3e} | "
              f"|db|={abs(ours.bias - float(sk.intercept_)):.3e}")


def truth_logistic(layer: int = 23, lam2: float = 0.03) -> None:
    section(f"3.3 truth（layer {layer}）逻辑回归：手写 GD vs sklearn LogisticRegression")
    xtr, ytr = load_split("truth", "train", layer)
    xva, yva = load_split("truth", "valid", layer)
    n = xtr.shape[0]
    C = 1.0 / (2.0 * n * lam2)
    print(f"换算：C = 1/(2*N*λ2) = {C:.6g}（N={n}，λ2={lam2}）")

    ours = LogisticRegressionModel(
        xtr.shape[1], lambda2=lam2, momentum=0.99, max_iter=8_000, tol=1e-12, patience=100
    )
    t0 = time.time()
    ours.fit(xtr, ytr)
    t_ours = time.time() - t0

    sk = SKLogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    t0 = time.time()
    sk.fit(xtr, ytr)
    t_sk = time.time() - t0

    sk_w = np.asarray(sk.coef_).ravel()
    acc_ours = np.mean((ours.predict(xva) >= 0.5) == yva)
    acc_sk = np.mean(sk.predict(xva) == yva)
    acc_ours_tr = np.mean((ours.predict(xtr) >= 0.5) == ytr)
    acc_sk_tr = np.mean(sk.predict(xtr) == ytr)
    cos = float(ours.weights @ sk_w / (np.linalg.norm(ours.weights) * np.linalg.norm(sk_w)))
    print(f"ours     trainAcc={acc_ours_tr:.4f} validAcc={acc_ours:.4f} "
          f"({t_ours:.1f}s, {ours.n_iter_} iters, NLL={ours.train_loss_:.5f})")
    print(f"sklearn  trainAcc={acc_sk_tr:.4f} validAcc={acc_sk:.4f} ({t_sk:.1f}s, "
          f"n_iter={int(sk.n_iter_.max())})")
    print(f"cos(w)={cos:.6f}  max|dw|={np.max(np.abs(ours.weights - sk_w)):.3e}  "
          f"bias diff={abs(ours.bias - float(sk.intercept_[0])):.3e}")
    # 概率层面的差异
    p_ours = ours.predict(xva)
    p_sk = sk.predict_proba(xva)[:, 1]
    print(f"验证集概率最大绝对差 = {np.max(np.abs(p_ours - p_sk)):.3e}，"
          f"预测类别不一致数 = {int(np.sum((p_ours >= 0.5) != (p_sk >= 0.5)))}/{len(yva)}")


def main() -> None:
    tiny_example()
    world_ridge()
    truth_logistic()


if __name__ == "__main__":
    main()
