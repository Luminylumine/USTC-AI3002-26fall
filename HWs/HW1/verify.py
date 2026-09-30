#!/usr/bin/env python3
"""HW1 手算结果的数值交叉校验（仅用于验证，不替代推导）。

运行：conda 环境 lsr-env-gpu 下  python verify.py
"""

import numpy as np

rng = np.random.default_rng(0)
PASS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    PASS.append(tag)
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""))


def close(a, b, tol=1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=tol, rtol=tol))


# ---------------------------------------------------------------- 题 1
def q1() -> None:
    print("\n=== 题 1：偏差-方差分解 ===")
    n_s, n_e = 400_000, 1  # 每个 S 抽一个 epsilon（S 与 epsilon 独立）
    # 构造一个玩具设定：phi(x0)=1，w_hat_S ~ N(1, 0.5)，f*(x0)=0.5，eps ~ N(0, sigma^2)
    sigma2, f_star, m_true, var_true = 0.7, 0.5, 1.0, 0.25
    w = rng.normal(loc=m_true, scale=np.sqrt(var_true), size=n_s)
    z = w * 1.0
    eps = rng.normal(0.0, np.sqrt(sigma2), size=n_s * n_e).reshape(n_s, n_e)
    y = f_star + eps
    mse = np.mean((y - z[:, None]) ** 2)
    rhs = sigma2 + (m_true - f_star) ** 2 + var_true
    check("E[(y-Z)^2] == sigma^2 + bias^2 + Var", abs(mse - rhs) < 0.02,
          f"MC={mse:.4f}  分解={rhs:.4f} (sigma^2={sigma2}, bias^2={0.5**2}, Var={var_true})")


# ---------------------------------------------------------------- 题 3
def q3() -> None:
    print("\n=== 题 3：线性回归 ===")
    Phi = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1], [1, -1, 0], [1, 0, -1], [1, -1, -1]], dtype=np.float64)
    y = np.array([2, 0, 3, -4, -2, -5], dtype=np.float64)
    P = np.diag([0.0, 1.0, 1.0])
    N = 6

    print(f"Phi.shape={Phi.shape}, y.shape={y.shape}, rank={np.linalg.matrix_rank(Phi)}")
    check("Phi^T Phi = [[6,0,0],[0,4,2],[0,2,4]]",
          close(Phi.T @ Phi, [[6, 0, 0], [0, 4, 2], [0, 2, 4]]), str((Phi.T @ Phi).tolist()))
    check("Phi^T y = (-6,14,10)", close(Phi.T @ y, [-6, 14, 10]), str((Phi.T @ y).tolist()))
    check("y^T y = 58", close(y @ y, 58))

    # (a) w_A 精确拟合
    wA = np.linalg.lstsq(Phi, y, rcond=None)[0]
    check("w_A = (-1,3,1) 且残差为 0", close(wA, [-1, 3, 1]) and close(Phi @ wA, y),
          f"w_A={np.round(wA, 12).tolist()}, max|resid|={np.max(np.abs(Phi @ wA - y)):.2e}")

    def J(w, lam):
        return np.mean((Phi @ w - y) ** 2) + lam * (w @ P @ w)

    def grad(w, lam):
        return (2.0 / N) * Phi.T @ (Phi @ w - y) + 2 * lam * P @ w

    # (c) 一步 GD
    lam, eta = 1 / 3, 1 / 4
    w0 = np.array([0.0, 1.0, -1.0])
    r = Phi @ w0 - y
    check("残差 r = (-1,-1,-3,3,3,5)", close(r, [-1, -1, -3, 3, 3, 5]), str(r.tolist()))
    gL = (2.0 / N) * Phi.T @ r
    check("grad L(w0) = (2,-4,-4)", close(gL, [2, -4, -4]), str(gL.tolist()))
    g = grad(w0, lam)
    check("grad J(w0) = (2,-10/3,-14/3)", close(g, [2, -10 / 3, -14 / 3]), str(np.round(g, 10).tolist()))
    w1 = w0 - eta * g
    check("一步后 w1 = (-1/2, 11/6, 1/6)", close(w1, [-0.5, 11 / 6, 1 / 6]), str(np.round(w1, 10).tolist()))
    check("J(w0) = 29/3", close(J(w0, lam), 29 / 3), f"{J(w0, lam):.10f} vs {29/3:.10f}")
    check("J(w1) = 367/108", close(J(w1, lam), 367 / 108), f"{J(w1, lam):.10f} vs {367/108:.10f}")
    check("J(w1) < J(w0) 下降", J(w1, lam) < J(w0, lam))

    # (d) 两个候选模型
    wB = np.array([-1.0, 2.0, 1.0])
    mseA, mseB = np.mean((Phi @ wA - y) ** 2), np.mean((Phi @ wB - y) ** 2)
    check("train MSE(w_A)=0, train MSE(w_B)=2/3", close(mseA, 0) and close(mseB, 2 / 3),
          f"{mseA:.6f}, {mseB:.6f}")
    check("||P w_A||^2=10, ||P w_B||^2=5", close(wA @ P @ wA, 10) and close(wB @ P @ wB, 5),
          f"{wA @ P @ wA}, {wB @ P @ wB}")

    Vx = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    Vy = np.array([1.0, 2.0, 1.0])
    Vphi = np.column_stack([np.ones(3), Vx])
    valA = np.mean((Vphi @ wA - Vy) ** 2)
    valB = np.mean((Vphi @ wB - Vy) ** 2)
    check("valid MSE(w_A)=5/3 < valid MSE(w_B)=2", close(valA, 5 / 3) and close(valB, 2),
          f"{valA:.6f} vs {valB:.6f}")

    # (f) c = 1 - 2 eta lambda，等价性：GD on J  ==  weight decay + grad L
    c = 1 - 2 * eta * lam
    Dc = np.diag([1.0, c, c])
    w_wd = Dc @ w0 - eta * ((2.0 / N) * Phi.T @ (Phi @ w0 - y))
    check("c = 1 - 2*eta*lambda = 5/6", close(c, 5 / 6), f"c={c:.10f}")
    check("权重衰减更新 == 对 J 的一步 GD", close(w_wd, w1), f"{np.round(w_wd, 10).tolist()} vs {np.round(w1, 10).tolist()}")

    # (e) lambda 增大时：正则范数不增、训练 MSE 不降（用闭式解数值验证）
    def closed_form(lam_):
        A = (Phi.T @ Phi) / N + lam_ * P
        b = (Phi.T @ y) / N
        return np.linalg.solve(A, b)

    w_l1, w_l2 = closed_form(0.0), closed_form(1.0)
    check("lambda 增大：||P w||^2 不增", w_l2 @ P @ w_l2 <= w_l1 @ P @ w_l1 + 1e-12,
          f"{w_l1 @ P @ w_l1:.4f} -> {w_l2 @ P @ w_l2:.4f}")
    check("lambda 增大：训练 MSE 不降",
          np.mean((Phi @ w_l2 - y) ** 2) >= np.mean((Phi @ w_l1 - y) ** 2) - 1e-12,
          f"{np.mean((Phi @ w_l1 - y)**2):.4f} -> {np.mean((Phi @ w_l2 - y)**2):.4f}")


# ---------------------------------------------------------------- 题 4
def q4() -> None:
    print("\n=== 题 4：逻辑回归 ===")
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))

    def nll(t, xs, ys):
        t = np.asarray(t, dtype=np.float64)
        p = sig(t[..., None] * xs)
        return -np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p), axis=-1)

    # A: x=-1 -> 3 个 0 / 1 个 1；x=1 -> 1 个 0 / 3 个 1
    xA = np.array([-1.0] * 4 + [1.0] * 4)
    yA = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.float64)
    # B: x=-1 -> 4 个 0；x=1 -> 4 个 1
    xB = np.array([-1.0] * 4 + [1.0] * 4)
    yB = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)

    ts = np.linspace(-6, 6, 24001)
    JA_closed = np.logaddexp(0.0, ts) - 0.75 * ts
    JB_closed = np.log1p(np.exp(-ts))
    check("J_A(t) = ln(1+e^t) - (3/4)t", close(nll(ts, xA, yA), JA_closed, 1e-8))
    check("J_B(t) = ln(1+e^{-t})", close(nll(ts, xB, yB), JB_closed, 1e-8))

    # 似然值
    t = 0.7
    LA = np.prod(sig(t * xA) ** yA * (1 - sig(t * xA)) ** (1 - yA))
    check("L_A(t) = sigma(t)^6 sigma(-t)^2", close(LA, sig(t) ** 6 * sig(-t) ** 2), f"{LA:.10f}")
    LB = np.prod(sig(t * xB) ** yB * (1 - sig(t * xB)) ** (1 - yB))
    check("L_B(t) = sigma(t)^8", close(LB, sig(t) ** 8), f"{LB:.10f}")

    # (b) A 的唯一最优 t=0；B 无有限最优
    tA = ts[np.argmin(nll(ts, xA, yA))]
    check("A 的唯一最优 t = ln3（数值网格）", abs(tA - np.log(3)) < 1e-3, f"t*≈{tA:.6f}  ln3={np.log(3):.6f}")
    dA = np.gradient(nll(ts, xA, yA), ts)
    check("A: J_A'' > 0（严格凸）", np.min(np.gradient(dA, ts)) > 0)
    tB = ts[np.argmin(nll(ts, xB, yB))]
    check("B 的下确界=0 且在网格最右端取到（无有限最优）",
          tB > 5.9 and nll(np.array([tB]), xB, yB)[0] > 0,
          f"网格最优 t={tB:.3f}, J={nll(np.array([tB]), xB, yB)[0]:.6f} > 0")
    check("B: J_B 单调递减", np.all(np.diff(nll(ts, xB, yB)) < 1e-12))

    # (c) t>0, c>1 放大
    t0, c = 1.0, 3.0
    pred = lambda t_: (t_ * np.array([-1.0, 1.0]) >= 0).astype(int)
    check("决策边界不变（预测类别不变）", np.array_equal(pred(t0), pred(c * t0)),
          f"{pred(t0).tolist()} -> {pred(c*t0).tolist()}（边界 x=0）")
    err = lambda t_: np.mean(pred(t_) != yB[[0, 4]])
    errA, errB = err(t0), err(c * t0)
    check("B 上错误率不变且为 0", close(errA, 0) and close(errB, 0), f"{errA} -> {errB}")
    p1 = sig(t0 * xB)
    p1_c = sig(c * t0 * xB)
    p_true = yB * p1 + (1 - yB) * (1 - p1)
    p_true_c = yB * p1_c + (1 - yB) * (1 - p1_c)
    check("真标签预测概率上升", np.all(p_true_c > p_true),
          f"mean {p_true.mean():.6f} -> {p_true_c.mean():.6f}")
    check("B 的平均 NLL 下降", nll(c * t0, xB, yB) < nll(t0, xB, yB),
          f"{nll(t0, xB, yB):.6f} -> {nll(c*t0, xB, yB):.6f}")

    # (d) 正则目标存在有限最小值
    lam = 0.1
    reg = lambda t_: nll(t_, xB, yB) + lam * t_ ** 2
    t_opt = ts[np.argmin(reg(ts))]
    check("J_B(t)+lambda t^2 存在有限最优（抑制权重发散）", -6 < t_opt < 6,
          f"t_opt≈{t_opt:.3f}（lambda={lam}）")


def main() -> None:
    q1()
    q3()
    q4()
    n_fail = PASS.count("FAIL")
    print(f"\n合计 {len(PASS)} 项，FAIL {n_fail} 项")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
