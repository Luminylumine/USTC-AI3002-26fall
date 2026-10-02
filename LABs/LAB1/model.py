"""线性回归与逻辑回归模型（手写全批量梯度下降，含 L1 次梯度与 L2 正则）。

实现要点（对应实验文档 2.1 节）：
- 损失、梯度均按文档公式手写，只用 numpy，不调用 sklearn 的拟合接口；
- 偏置 b 不参与 L1/L2 正则；
- L1 在零点取次梯度 sgn(0)=0；
- 学习率默认由幂迭代估计的 Lipschitz 常数自动给出（lr = 1/L），
  Lipschitz 常数基于含全 1 偏置列的增广设计矩阵估计，覆盖偏置坐标，
  避免手调学习率在不同数据集/层数上失效；
- 支持学习率衰减与动量（二者都不改变"梯度下降法"这一事实），
  用于缓解文档提示的"含 L1 时固定学习率在零点附近振荡"；
- predict 对线性模型返回预测值，对逻辑模型返回标签为 1 的概率（不做阈值化）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["LinearRegressionModel", "LogisticRegressionModel"]


def _as_2d(features: np.ndarray) -> np.ndarray:
    """把输入整理成 float64 的二维设计矩阵 (N, D)。"""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"features 应为二维数组，实际形状 {x.shape}")
    return x


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid：对 z>=0 用 1/(1+e^-z)，对 z<0 用 e^z/(1+e^z)。"""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def _lambda_max(x: np.ndarray, n_iter: int = 100) -> float:
    """用幂迭代估计 (1/N) X_aug^T X_aug 的最大特征值（用于自动设置学习率）。

    这里 X_aug = [X, 1] 是在原始特征后拼接全 1 偏置列后的增广设计矩阵，
    使估计出的 Lipschitz 常数覆盖偏置坐标。否则当特征为常数（尤其全零）时
    X^T X 的最大特征值为 0，学习率会错误回退到 1 而振荡/发散。
    拼接偏置列后，全零特征时最大特征值至少为 1（偏置列的单位范数），
    保证默认全批量 GD 的学习率有限且收敛。
    """
    n, n_features = x.shape
    v = np.ones(n_features + 1, dtype=np.float64) / np.sqrt(n_features + 1)
    lam = 0.0
    for _ in range(n_iter):
        u = x @ v[:n_features] + v[n_features]
        nv = np.linalg.norm(u)
        if nv == 0.0:
            return 0.0
        u /= nv
        v = np.empty(n_features + 1, dtype=np.float64)
        v[:n_features] = x.T @ u
        v[n_features] = float(np.sum(u))
        nv = np.linalg.norm(v)
        if nv == 0.0:
            return 0.0
        v /= nv
        # Rayleigh 商 v^T (X_aug^T X_aug) v / N
        xu = x @ v[:n_features] + v[n_features]
        lam = float(
            (v[:n_features] @ (x.T @ xu) + v[n_features] * np.sum(xu)) / max(n, 1)
        )
    return max(lam, 0.0)


class _GradientDescentBase:
    """两个模型共享的梯度下降训练流程。"""

    # 子类通过这两个方法给出数据损失与其梯度
    def _data_loss_and_grad(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
    ) -> tuple[float, np.ndarray, float]:
        raise NotImplementedError

    def __init__(
        self,
        n_features: int,
        *,
        lambda1: float = 0.0,
        lambda2: float = 0.0,
        learning_rate: float | None = None,
        lr_scale: float = 1.0,
        lr_decay: float = 0.0,
        momentum: float = 0.0,
        max_iter: int = 20_000,
        tol: float = 1e-10,
        patience: int = 50,
        check_every: int = 10,
        verbose: bool = False,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features 必须为正")
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("正则化强度不能为负")
        self.n_features = int(n_features)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.learning_rate = learning_rate
        self.lr_scale = float(lr_scale)
        self.lr_decay = float(lr_decay)
        self.momentum = float(momentum)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.patience = int(patience)
        self.check_every = max(int(check_every), 1)
        self.verbose = bool(verbose)

        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = np.float64(0.0)
        self.n_iter_: int = 0
        self.train_loss_: float = np.inf
        self.loss_curve_: list[tuple[int, float]] = []

    # ---------------------------------------------------------- 损失与正则
    def _regularizer(self, weights: np.ndarray) -> float:
        """lambda1*||w||_1 + lambda2*||w||_2^2（偏置不参与正则）。"""
        return self.lambda1 * np.sum(np.abs(weights)) + self.lambda2 * float(weights @ weights)

    def _regularizer_grad(self, weights: np.ndarray) -> np.ndarray:
        """L1 次梯度 sgn(w)（sgn(0)=0）与 L2 梯度 2*lambda2*w。"""
        return self.lambda1 * np.sign(weights) + 2.0 * self.lambda2 * weights

    def _initial_learning_rate(self, x: np.ndarray) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        # 线性：Hessian(数据项) = (2/N) X^T X ；逻辑：Hessian ⪯ (1/4)(1/N) X^T X
        raise NotImplementedError

    # ---------------------------------------------------------- 训练主循环
    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = _as_2d(features)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"样本数不一致：features {x.shape[0]} vs targets {y.shape[0]}")
        if x.shape[1] != self.n_features:
            raise ValueError(f"特征维度不一致：期望 {self.n_features}，实际 {x.shape[1]}")

        weights = np.zeros(x.shape[1], dtype=np.float64)
        bias = 0.0
        velocity = np.zeros_like(weights)
        lr0 = self._initial_learning_rate(x)

        best_loss = np.inf
        stall = 0
        self.loss_curve_.clear()

        for it in range(1, self.max_iter + 1):
            lr_t = lr0 / (1.0 + self.lr_decay * (it - 1))

            _, grad_data_w, grad_data_b = self._data_loss_and_grad(x, y, weights, bias)

            grad_w = grad_data_w + self._regularizer_grad(weights)
            grad_b = grad_data_b  # 偏置无正则项

            # 动量（可选）+ 全批量梯度下降，所有参数同时更新
            velocity = self.momentum * velocity - lr_t * grad_w
            weights = weights + velocity
            bias = bias - lr_t * grad_b
            self.n_iter_ = it

            if it % self.check_every == 0:
                new_loss = self._data_loss_and_grad(x, y, weights, bias)[0] + self._regularizer(
                    weights
                )
                self.loss_curve_.append((it, new_loss))
                if not np.isfinite(new_loss):
                    raise FloatingPointError(f"第 {it} 步损失发散（lr={lr_t:.3g}），请调小 learning_rate")
                # 相对改进小于 tol 记一次"停滞"
                if new_loss > best_loss - self.tol * max(abs(best_loss), 1.0):
                    stall += 1
                else:
                    stall = 0
                    best_loss = min(best_loss, new_loss)
                if self.verbose and (it % (self.check_every * 20) == 0):
                    print(f"  iter {it:>6d}  loss {new_loss:.10f}  lr {lr_t:.3g}")
                if stall >= self.patience:
                    break

        self.weights = weights
        self.bias = np.float64(bias)
        self.train_loss_ = float(
            self._data_loss_and_grad(x, y, weights, bias)[0] + self._regularizer(weights)
        )


class LinearRegressionModel(_GradientDescentBase):
    """用单个线性回归模型预测一个连续目标。"""

    def __init__(self, n_features: int, **kwargs) -> None:
        super().__init__(n_features, **kwargs)

    def _initial_learning_rate(self, x: np.ndarray) -> float:
        # L = lambda_max((2/N) X_aug^T X_aug) + 2*lambda2（X_aug 含偏置列）
        l_max = _lambda_max(x)
        lipschitz = 2.0 * l_max + 2.0 * self.lambda2
        if lipschitz <= 0:
            return 1.0
        return self.lr_scale / lipschitz

    def _data_loss_and_grad(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
    ) -> tuple[float, np.ndarray, float]:
        n = x.shape[0]
        residual = x @ weights + bias - y
        loss = float(np.mean(residual**2))
        grad_w = (2.0 / n) * (x.T @ residual)
        grad_b = float((2.0 / n) * np.sum(residual))
        return loss, grad_w, grad_b

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = _as_2d(features)
        if x.shape[1] != self.n_features:
            raise ValueError(f"特征维度不一致：期望 {self.n_features}，实际 {x.shape[1]}")
        return x @ self.weights + self.bias


class LogisticRegressionModel(_GradientDescentBase):
    """用逻辑回归从隐藏表征预测句子真假。"""

    def __init__(self, n_features: int, **kwargs) -> None:
        super().__init__(n_features, **kwargs)

    def _initial_learning_rate(self, x: np.ndarray) -> float:
        # 逻辑回归的 Hessian <= (1/4)(1/N) X_aug^T X_aug（X_aug 含偏置列）
        l_max = _lambda_max(x)
        lipschitz = 0.25 * l_max + 2.0 * self.lambda2
        if lipschitz <= 0:
            return 1.0
        return self.lr_scale / lipschitz

    def _data_loss_and_grad(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
    ) -> tuple[float, np.ndarray, float]:
        n = x.shape[0]
        z = x @ weights + bias
        # 平均负对数似然：mean( logaddexp(0, z) - y*z )
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        p = _sigmoid(z)
        grad_w = (1.0 / n) * (x.T @ (p - y))
        grad_b = float((1.0 / n) * np.sum(p - y))
        return loss, grad_w, grad_b

    def predict(self, features: np.ndarray) -> np.ndarray:
        """返回标签为 1 的概率，形状 (N,)，不做 0.5 阈值化。"""
        x = _as_2d(features)
        if x.shape[1] != self.n_features:
            raise ValueError(f"特征维度不一致：期望 {self.n_features}，实际 {x.shape[1]}")
        return _sigmoid(x @ self.weights + self.bias)
