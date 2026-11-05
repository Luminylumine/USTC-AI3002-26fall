#!/usr/bin/env python3
"""HW2 手算结果与推导的数值交叉校验（仅用于验证，不替代推导）。

运行：conda 环境 lsr-env-gpu 下  python HWs/HW2/verify.py
"""

import itertools
import numpy as np
import torch

rng = np.random.default_rng(42)
torch.manual_seed(42)
PASS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    PASS.append(tag)
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""))


def close(a, b, tol=1e-7) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=tol, rtol=tol))


# ---------------------------------------------------------------- 题 1
def q1() -> None:
    print("\n=== 题 1：神经网络的前向传播与反向传播 ===")
    # (a) 前向传播
    x = np.array([1.0, -1.0])
    W = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([-1.0, 0.0])
    v = np.array([1.0, -1.0])
    y = 1.0

    z = W.T @ x + b
    check("z = [0, 0]", close(z, [0.0, 0.0]), f"z={z.tolist()}")

    def sigma(t):
        return 1.0 / (1.0 + np.exp(-t))

    h = sigma(z)
    check("h = [0.5, 0.5]", close(h, [0.5, 0.5]), f"h={h.tolist()}")

    y_hat = float(v.T @ h)
    check("y_hat = 0.0", close(y_hat, 0.0), f"y_hat={y_hat}")

    loss = 0.5 * (y_hat - y) ** 2
    check("L = 0.5", close(loss, 0.5), f"L={loss}")

    # (b) 链式法则梯度推导与数值
    # 手算解析值
    # dL/dv = (y_hat - y) * h
    grad_v_manual = (y_hat - y) * h  # [-0.5, -0.5]
    check("grad_v = [-0.5, -0.5]", close(grad_v_manual, [-0.5, -0.5]), f"grad_v={grad_v_manual.tolist()}")

    # delta = dL/dz = (y_hat - y) * v * sigma'(z)
    # sigma'(0) = 0.25
    sigma_prime_z = h * (1.0 - h)
    delta = (y_hat - y) * v * sigma_prime_z  # [-0.25, 0.25]
    grad_b_manual = delta
    check("grad_b = [-0.25, 0.25]", close(grad_b_manual, [-0.25, 0.25]), f"grad_b={grad_b_manual.tolist()}")

    # dL/dW = x @ delta^T
    grad_W_manual = np.outer(x, delta)
    expected_grad_W = np.array([[-0.25, 0.25], [0.25, -0.25]])
    check("grad_W = [[-0.25, 0.25], [0.25, -0.25]]", close(grad_W_manual, expected_grad_W),
          f"\n{grad_W_manual}")

    # 使用 PyTorch Autograd 双向核对
    xt = torch.tensor([1.0, -1.0], dtype=torch.float64)
    Wt = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    bt = torch.tensor([-1.0, 0.0], dtype=torch.float64, requires_grad=True)
    vt = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(1.0, dtype=torch.float64)

    zt = Wt.T @ xt + bt
    ht = torch.sigmoid(zt)
    y_hatt = vt @ ht
    Lt = 0.5 * (y_hatt - yt) ** 2
    Lt.backward()

    check("Autograd 核对 grad_v", close(grad_v_manual, vt.grad.numpy()), f"{vt.grad.numpy().tolist()}")
    check("Autograd 核对 grad_b", close(grad_b_manual, bt.grad.numpy()), f"{bt.grad.numpy().tolist()}")
    check("Autograd 核对 grad_W", close(grad_W_manual, Wt.grad.numpy()), f"\n{Wt.grad.numpy()}")

    # 有限差分双向核对
    eps = 1e-6

    def eval_loss(W_val, b_val, v_val):
        zv = W_val.T @ x + b_val
        hv = sigma(zv)
        yv = float(v_val.T @ hv)
        return 0.5 * (yv - y) ** 2

    # 有限差分核对 grad_v
    num_grad_v = np.zeros_like(v)
    for i in range(2):
        vp = v.copy()
        vp[i] += eps
        vm = v.copy()
        vm[i] -= eps
        num_grad_v[i] = (eval_loss(W, b, vp) - eval_loss(W, b, vm)) / (2 * eps)
    check("有限差分核对 grad_v", close(grad_v_manual, num_grad_v, tol=1e-5))

    # 有限差分核对 grad_b
    num_grad_b = np.zeros_like(b)
    for i in range(2):
        bp = b.copy()
        bp[i] += eps
        bm = b.copy()
        bm[i] -= eps
        num_grad_b[i] = (eval_loss(W, bp, v) - eval_loss(W, bm, v)) / (2 * eps)
    check("有限差分核对 grad_b", close(grad_b_manual, num_grad_b, tol=1e-5))

    # 有限差分核对 grad_W
    num_grad_W = np.zeros_like(W)
    for i in range(2):
        for j in range(2):
            Wp = W.copy()
            Wp[i, j] += eps
            Wm = W.copy()
            Wm[i, j] -= eps
            num_grad_W[i, j] = (eval_loss(Wp, b, v) - eval_loss(Wm, b, v)) / (2 * eps)
    check("有限差分核对 grad_W", close(grad_W_manual, num_grad_W, tol=1e-5))

    # (c)(i) 随机采样验证 prod sigma'(z_l) <= 4^{-m}
    all_bounded = True
    for m in [1, 2, 3, 5, 10]:
        z_samples = rng.normal(0, 5, size=(10000, m))
        s_prime = sigma(z_samples) * (1.0 - sigma(z_samples))
        prod_val = np.prod(s_prime, axis=1)
        bound = 4.0 ** (-m)
        if not np.all(prod_val <= bound + 1e-12):
            all_bounded = False
        # 在 z=0 处验证取等
        z_zero = np.zeros(m)
        prod_zero = np.prod(sigma(z_zero) * (1.0 - sigma(z_zero)))
        check(f"m={m} 时 prod sigma'(0) == 4^(-{m})", close(prod_zero, bound))
    check("所有随机采样均满足 |prod sigma'(z_l)| <= 4^(-m)", all_bounded)


# ---------------------------------------------------------------- 题 2
def q2() -> None:
    print("\n=== 题 2：K-Means 目标下降与解的质量 ===")
    # (a) 验证均值最小化平方误差和
    for _ in range(5):
        Ck = rng.normal(size=(20, 3))
        x_bar = np.mean(Ck, axis=0)
        base_err = np.sum((Ck - x_bar) ** 2)
        u_rand = rng.normal(size=3)
        u_err = np.sum((Ck - u_rand) ** 2)
        check("sum ||x_i - x_bar||^2 <= sum ||x_i - u||^2", u_err >= base_err - 1e-12)

    # (c) 验证 A 和 B 均为不动点
    X = np.array([[-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, 1.0]])

    # 状态 A：左右分簇，中心 (-2, 0), (2, 0)
    mu_A = np.array([[-2.0, 0.0], [2.0, 0.0]])
    dist_A = np.sum((X[:, None, :] - mu_A[None, :, :]) ** 2, axis=2)  # shape (4, 2)
    assign_A = np.argmin(dist_A, axis=1)
    check("状态 A 分配结果为 [0, 0, 1, 1]", np.array_equal(assign_A, [0, 0, 1, 1]), f"{assign_A.tolist()}")
    mu_A_next = np.array([np.mean(X[assign_A == k], axis=0) for k in range(2)])
    check("状态 A 一次迭代后中心不变（不动点）", close(mu_A_next, mu_A), f"{mu_A_next.tolist()}")
    J_A = np.sum(np.min(dist_A, axis=1))
    check("目标值 J_A = 4", close(J_A, 4.0), f"J_A={J_A}")

    # 状态 B：上下分簇，中心 (0, -1), (0, 1)
    mu_B = np.array([[0.0, -1.0], [0.0, 1.0]])
    dist_B = np.sum((X[:, None, :] - mu_B[None, :, :]) ** 2, axis=2)
    assign_B = np.argmin(dist_B, axis=1)
    check("状态 B 分配结果为 [0, 1, 0, 1]", np.array_equal(assign_B, [0, 1, 0, 1]), f"{assign_B.tolist()}")
    mu_B_next = np.array([np.mean(X[assign_B == k], axis=0) for k in range(2)])
    check("状态 B 一次迭代后中心不变（不动点）", close(mu_B_next, mu_B), f"{mu_B_next.tolist()}")
    J_B = np.sum(np.min(dist_B, axis=1))
    check("目标值 J_B = 16", close(J_B, 16.0), f"J_B={J_B}")

    # (d) 验证 J_{K+1}^* <= J_K^* 以及 J_n^* = 0
    # 对 X 数据（4个点）穷举所有可能的分簇计算精确最优 J_1^*, J_2^*, J_3^*, J_4^*
    n_pts = len(X)

    def exact_J_star(K_val):
        best_j = float("inf")
        for partition in itertools.product(range(K_val), repeat=n_pts):
            j_val = 0.0
            for k in range(K_val):
                idx = [i for i, c in enumerate(partition) if c == k]
                if len(idx) > 0:
                    pts_k = X[idx]
                    mean_k = np.mean(pts_k, axis=0)
                    j_val += np.sum((pts_k - mean_k) ** 2)
            if j_val < best_j:
                best_j = j_val
        return best_j

    J1_star = exact_J_star(1)
    J2_star = exact_J_star(2)
    J3_star = exact_J_star(3)
    J4_star = exact_J_star(4)
    check("精确全局最优 J_1^* >= J_2^* >= J_3^* >= J_4^*",
          J1_star >= J2_star >= J3_star >= J4_star - 1e-12,
          f"J1*={J1_star:.1f}, J2*={J2_star:.1f}, J3*={J3_star:.1f}, J4*={J4_star:.1f}")
    check("J_n^* = 0（当 K=n 时）", close(J4_star, 0.0), f"J_4*={J4_star}")


# ---------------------------------------------------------------- 题 3
def q3() -> None:
    print("\n=== 题 3：K-Means++ 距离、样本数与离群点 ===")
    # (a) 下界验证
    # 设 m=100, s=2, r=1, R=5
    m, s, r, R = 100, 2, 1.0, 5.0
    for _ in range(5):
        # A 样本 D^2 <= r^2
        D2_A = rng.uniform(0.01, r ** 2, size=m)
        # B 样本 D^2 >= R^2
        D2_B = rng.uniform(R ** 2, 2 * R ** 2, size=s)
        P_B_exact = np.sum(D2_B) / (np.sum(D2_A) + np.sum(D2_B))
        bound = (s * R ** 2) / (m * r ** 2 + s * R ** 2)
        check("P(B) >= s*R^2 / (m*r^2 + s*R^2)", P_B_exact >= bound - 1e-12,
              f"P(B)={P_B_exact:.4f} >= {bound:.4f}")

    # (b)(i) 概率公式与对比
    m_val = 3
    R_val = 4.0  # R > 2
    # 尚未选中的样本：m 个在 1, m 个在 2, 1 个在 R
    # 中心在 0，故 D(1)=1, D(2)=2, D(R)=R
    D2_list = [1.0 ** 2] * m_val + [2.0 ** 2] * m_val + [R_val ** 2]
    prob_exact = (R_val ** 2) / sum(D2_list)
    formula_prob = (R_val ** 2) / (5 * m_val + R_val ** 2)
    check("离群点概率公式 P_D2(R) = R^2 / (5m + R^2)", close(prob_exact, formula_prob),
          f"exact={prob_exact:.6f}, formula={formula_prob:.6f}")

    prob_uniform = 1.0 / (2 * m_val + 1)
    check("P_D2(R) > 1 / (2m + 1)（当 R > 2 时）", prob_exact > prob_uniform,
          f"P_D2={prob_exact:.4f} > P_uniform={prob_uniform:.4f}")

    # 具体算例核对：m=2, R=4
    m_ex, R_ex = 2, 4.0
    P_ex = (R_ex ** 2) / (5 * m_ex + R_ex ** 2)  # 16 / (10 + 16) = 16 / 26 = 8 / 13
    check("算例 m=2, R=4 处 P_D2 = 8/13", close(P_ex, 8 / 13), f"P={P_ex:.6f} vs 8/13={8/13:.6f}")

    # (b)(ii) 当 R^2 >= 5m 时，P >= 1/2
    R_bound = np.sqrt(5 * m_val)
    prob_at_bound = (R_bound ** 2) / (5 * m_val + R_bound ** 2)
    check("R^2 = 5m 时，P_D2(R) == 0.5", close(prob_at_bound, 0.5), f"P={prob_at_bound}")
    prob_above = ((R_bound + 1) ** 2) / (5 * m_val + (R_bound + 1) ** 2)
    check("R^2 > 5m 时，P_D2(R) >= 0.5", prob_above >= 0.5, f"P={prob_above:.4f}")


# ---------------------------------------------------------------- 题 4
def q4() -> None:
    print("\n=== 题 4：GMM 与 EM ===")

    # (b) M 步推导验证：检查一个 2D GMM 上执行一次 M 步后 Q(theta | theta^(t)) 是否极大化
    d, K, n = 2, 2, 30
    X = rng.normal(size=(n, d))
    # 随机生成合法的责任度 gamma (n, K)
    gamma_raw = rng.uniform(0.1, 1.0, size=(n, K))
    gamma = gamma_raw / np.sum(gamma_raw, axis=1, keepdims=True)
    N_k = np.sum(gamma, axis=0)

    # 按推导公式进行 M 步更新
    pi_new = N_k / n
    mu_new = (gamma.T @ X) / N_k[:, None]  # (K, d)
    Sigma_new = np.zeros((K, d, d))
    for k in range(K):
        diff = X - mu_new[k]  # (n, d)
        Sigma_new[k] = (diff.T @ (gamma[:, k:k + 1] * diff)) / N_k[k]

    def compute_Q(pi_val, mu_val, Sigma_val):
        Q_val = 0.0
        for k in range(K):
            sign, logdet = np.linalg.slogdet(Sigma_val[k])
            inv_sig = np.linalg.inv(Sigma_val[k])
            diff = X - mu_val[k]
            quad = np.sum((diff @ inv_sig) * diff, axis=1)  # (n,)
            log_p = np.log(pi_val[k]) - 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * quad
            Q_val += np.sum(gamma[:, k] * log_p)
        return Q_val

    Q_opt = compute_Q(pi_new, mu_new, Sigma_new)

    # 随机扰动 mu 验证 Q 下降
    mu_perturbed = mu_new + rng.normal(scale=0.05, size=mu_new.shape)
    Q_pert_mu = compute_Q(pi_new, mu_perturbed, Sigma_new)
    check("M 步所得 mu 最大化 Q（扰动后 Q 下降）", Q_opt >= Q_pert_mu,
          f"Q_opt={Q_opt:.4f} >= Q_pert={Q_pert_mu:.4f}")

    # (c) ELBO 不等式 ell(theta) >= L(q, theta)
    def compute_log_lik(pi_val, mu_val, Sigma_val):
        lik = np.zeros(n)
        for k in range(K):
            sign, logdet = np.linalg.slogdet(Sigma_val[k])
            inv_sig = np.linalg.inv(Sigma_val[k])
            diff = X - mu_val[k]
            quad = np.sum((diff @ inv_sig) * diff, axis=1)
            pdf = (1.0 / np.sqrt((2 * np.pi) ** d * np.exp(logdet))) * np.exp(-0.5 * quad)
            lik += pi_val[k] * pdf
        return np.sum(np.log(lik))

    ell_val = compute_log_lik(pi_new, mu_new, Sigma_new)
    # 计算此时以真实后验为 q 时的 ELBO
    gamma_exact = np.zeros((n, K))
    for k in range(K):
        sign, logdet = np.linalg.slogdet(Sigma_new[k])
        inv_sig = np.linalg.inv(Sigma_new[k])
        diff = X - mu_new[k]
        quad = np.sum((diff @ inv_sig) * diff, axis=1)
        gamma_exact[:, k] = pi_new[k] * np.exp(-0.5 * quad) / np.sqrt((2 * np.pi) ** d * np.exp(logdet))
    gamma_exact = gamma_exact / np.sum(gamma_exact, axis=1, keepdims=True)

    def compute_ELBO(q_val):
        elbo = 0.0
        for k in range(K):
            sign, logdet = np.linalg.slogdet(Sigma_new[k])
            inv_sig = np.linalg.inv(Sigma_new[k])
            diff = X - mu_new[k]
            quad = np.sum((diff @ inv_sig) * diff, axis=1)
            log_joint = np.log(pi_new[k]) - 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * quad
            elbo += np.sum(q_val[:, k] * (log_joint - np.log(q_val[:, k] + 1e-15)))
        return elbo

    elbo_exact = compute_ELBO(gamma_exact)
    check("q 取真实后验时 ELBO 等于对数似然 ell(theta)", close(elbo_exact, ell_val, tol=1e-6),
          f"ELBO={elbo_exact:.6f}, ell={ell_val:.6f}")

    # 随机扰动 q
    q_pert = rng.uniform(0.1, 1.0, size=(n, K))
    q_pert = q_pert / np.sum(q_pert, axis=1, keepdims=True)
    elbo_pert = compute_ELBO(q_pert)
    check("任意辅助分布 q 满足 ELBO <= ell(theta)", elbo_pert < ell_val,
          f"ELBO_pert={elbo_pert:.6f} < ell={ell_val:.6f}")

    # (e) sigma^2 -> 0 时责任度与并列情况验证（数值稳定实现：减去 min distance 平移）
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([2.0, 0.0])
    # 唯一最近中心：点 (0.1, 0.0) 离 mu1 距离 0.1，离 mu2 距离 1.9
    x_unique = np.array([0.1, 0.0])
    pi_equal = np.array([0.5, 0.5])
    sigma2_small = 1e-4

    def resp_iso_stable(x_pt, pi_vec, s2):
        d1 = np.sum((x_pt - mu1) ** 2)
        d2 = np.sum((x_pt - mu2) ** 2)
        d_min = min(d1, d2)
        # 稳定 Softmax：指数上除以 2*s2 前先减去最小值，防止下溢为 0
        w1 = pi_vec[0] * np.exp(-(d1 - d_min) / (2 * s2))
        w2 = pi_vec[1] * np.exp(-(d2 - d_min) / (2 * s2))
        return w1 / (w1 + w2), w2 / (w1 + w2)

    r1, r2 = resp_iso_stable(x_unique, pi_equal, sigma2_small)
    check("唯一最近中心：sigma^2 -> 0 时 gamma_{ik*} -> 1, gamma_{ik} -> 0",
          r1 > 0.99999 and r2 < 1e-5, f"gamma=({r1:.6f}, {r2:.6f})")

    # 并列最近中心：点 (1.0, 0.0) 离 mu1 和 mu2 距离严格相等 (1.0)
    x_tied = np.array([1.0, 0.0])
    r1_tied, r2_tied = resp_iso_stable(x_tied, pi_equal, sigma2_small)
    check("并列最近中心（等权重）：sigma^2 -> 0 时责任度平分为 (0.5, 0.5)，不趋于 1",
          close(r1_tied, 0.5) and close(r2_tied, 0.5), f"gamma=({r1_tied:.4f}, {r2_tied:.4f})")

    # 并列最近中心（非等权重 pi = [0.3, 0.7]）：趋于先验比例
    pi_unequal = np.array([0.3, 0.7])
    r1_uneq, r2_uneq = resp_iso_stable(x_tied, pi_unequal, sigma2_small)
    check("并列最近中心（不等权 pi=[0.3, 0.7]）：sigma^2 -> 0 时趋于先验比例 (0.3, 0.7)",
          close(r1_uneq, 0.3) and close(r2_uneq, 0.7), f"gamma=({r1_uneq:.4f}, {r2_uneq:.4f})")


def main() -> None:
    print("开始 HW2 数值交叉校验...")
    q1()
    q2()
    q3()
    q4()
    n_fail = sum(1 for tag in PASS if tag != "PASS")
    print(f"\n合计 {len(PASS)} 项，FAIL {n_fail} 项")


if __name__ == "__main__":
    main()
