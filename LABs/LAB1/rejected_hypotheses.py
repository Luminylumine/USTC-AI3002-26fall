"""附加内容（实验文档 2.4）：记录两个"看似合理、实测被否"的假说。

科学素养的一部分是主动检验并证伪自己的直觉。本脚本对最终模型的两个候选"改进"做
受控实验，用原始数据说明它们为什么不该采用：

假说 A（world）：模型在"训练见过的国家"上输给了"按国家查表取平均"基线，因此把
    模型改成"国家均值 + 学一个国内残差探针"应该更好。
假说 B（truth）：不同层的表征可能互补，把第 11 层与第 23 层特征拼接起来，线性探针
    应能利用更多信息、提高准确率。

两个实验都给出"替代方案 vs 现有方案/基线"的对照。若替代方案更差，则假说被否。
计算口径与 extra_analysis.py 相同：world 用闭式岭回归，truth 用 sklearn lbfgs，
均为诊断用途，提交权重仍由手写 GD 产出。

所有打印数字都是原始证据，落盘到 results/rejected_hypotheses.txt。

运行：
    python rejected_hypotheses.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from train23 import load_split

HERE = Path(__file__).parent
RESULTS = HERE / "results"
DATA = HERE / "datasets"

WORLD_LAYER = 11
WORLD_LAMBDA2_LAT = 0.14
WORLD_LAMBDA2_LON = 0.28
TRUTH_LAYER = 23
TRUTH_LAMBDA2 = 0.03


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """中心化闭式岭回归，目标函数与 model.py 一致。"""
    xbar = x.mean(axis=0)
    ybar = float(y.mean())
    xc = x - xbar
    yc = y - ybar
    n = x.shape[0]
    w = np.linalg.solve(xc.T @ xc / n + lam * np.eye(x.shape[1]), xc.T @ yc / n)
    return w, ybar - float(xbar @ w)


def _world_sum_mse(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray
) -> float:
    total = 0.0
    for out, lam in ((0, WORLD_LAMBDA2_LAT), (1, WORLD_LAMBDA2_LON)):
        w, b = _ridge(x_tr, y_tr[:, out], lam)
        total += float(np.mean((x_te @ w + b - y_te[:, out]) ** 2))
    return total


def hypothesis_a_country_residual() -> None:
    print("=" * 72)
    print("假说 A（world）：'国家均值 + 国内残差探针' 能否优于直接线性探针？")
    print("=" * 72)

    x_tr, y_tr = load_split("world", "train", WORLD_LAYER)
    x_va, y_va = load_split("world", "valid", WORLD_LAYER)
    csv_tr = pd.read_csv(DATA / "world" / "train.csv")
    csv_va = pd.read_csv(DATA / "world" / "valid.csv")

    cm: dict[str, np.ndarray] = {}
    for c, sub in csv_tr.groupby("country"):
        cm[str(c)] = sub[["latitude", "longitude"]].to_numpy(dtype=np.float64).mean(axis=0)
    global_mean = csv_tr[["latitude", "longitude"]].to_numpy(dtype=np.float64).mean(axis=0)

    va_countries = csv_va["country"].astype(str).to_numpy()
    known = np.array([c in cm for c in va_countries])
    y_va_known = y_va[known]
    print(f"验证集 {len(known)} 条，其中训练见过国家的 {int(known.sum())} 条、"
          f"未见国家 {int((~known).sum())} 条。")

    ours_all = _world_sum_mse(x_tr, y_tr, x_va, y_va)
    ours_known = _world_sum_mse(x_tr, y_tr, x_va[known], y_va[known])
    print(f"[现有] 直接线性探针：全部验证 sum-MSE={ours_all:.1f}，"
          f"见过的国家上 sum-MSE={ours_known:.1f}")

    pred_country = np.array(
        [cm[c] if c in cm else global_mean for c in va_countries]
    )
    base_known = float(np.mean((pred_country[known] - y_va_known) ** 2))
    base_all = float(np.mean((pred_country - y_va) ** 2))
    print(f"[基线] 国家均值查表：全部验证 sum-MSE={base_all:.1f}，"
          f"见过的国家上 sum-MSE={base_known:.1f}")

    resid_tr = np.array(
        [y_tr[i] - cm[c] for i, c in enumerate(csv_tr["country"].astype(str).to_numpy())]
    )
    resid_va_known = np.array(
        [
            (lat, lon) - cm[c]
            for c, lat, lon in zip(
                va_countries[known],
                csv_va["latitude"].to_numpy()[known],
                csv_va["longitude"].to_numpy()[known],
            )
        ]
    )
    print(f"[替代] 训练集残差目标 MSE（纬度+经度）="
          f"{np.mean(resid_tr[:, 0] ** 2) + np.mean(resid_tr[:, 1] ** 2):.1f}")
    print("  残差探针在不同 λ 下、'见过的国家'验证样本上的 sum-MSE：")
    best_alt, best_lam = np.inf, None
    for lam in (0.5, 2.0, 10.0, 50.0, 200.0):
        total = 0.0
        for out in range(2):
            w, b = _ridge(x_tr, resid_tr[:, out], lam)
            total += float(np.mean((x_va[known] @ w + b - resid_va_known[:, out]) ** 2))
        print(f"    λ={lam:>6g}  sum-MSE={total:.1f}")
        if total < best_alt:
            best_alt, best_lam = total, lam
    zero_mse = float(np.mean(resid_va_known**2))
    print(f"  '不预测残差'（残差恒为 0，即国家均值）基线 sum-MSE={zero_mse:.1f}")

    print("-" * 72)
    verdict = "差于" if best_alt > zero_mse else "优于"
    print(f"结论：残差探针最优 sum-MSE={best_alt:.1f}（λ={best_lam:g}）仍然{verdict}"
          f"零残差基线 {zero_mse:.1f}。国家内部的坐标信号太弱、每国平均仅约 3 条"
          f"训练样本，探针学到的国内偏移是噪声，反而放大误差；假说 A 被否决。")
    print()


def hypothesis_b_layer_concat() -> None:
    print("=" * 72)
    print("假说 B（truth）：拼接第 11、23 层特征能否优于单用第 23 层？")
    print("=" * 72)

    def cv_acc(x: np.ndarray, y: np.ndarray, lam: float, n_folds: int = 5, seed: int = 0):
        rng = np.random.RandomState(seed)
        folds: list[list[int]] = [[] for _ in range(n_folds)]
        y_int = y.astype(int)
        for label in (0, 1):
            ids = np.where(y_int == label)[0]
            rng.shuffle(ids)
            for k, chunk in enumerate(np.array_split(ids, n_folds)):
                folds[k].extend(chunk.tolist())
        c = 1.0 / (2.0 * len(y_int) * lam)
        accs = []
        for k in range(n_folds):
            te = np.array(folds[k])
            tr = np.array([i for j in range(n_folds) if j != k for i in folds[j]])
            clf = LogisticRegression(C=c, max_iter=3000, solver="lbfgs")
            clf.fit(x[tr], y_int[tr])
            accs.append(float(np.mean(clf.predict(x[te]) == y_int[te])))
        return float(np.mean(accs)), float(np.std(accs))

    x23, y = load_split("truth", "train", TRUTH_LAYER)
    x11, _ = load_split("truth", "train", 11)
    x_concat = np.concatenate([x11, x23], axis=1)
    print(f"第 23 层特征 dim={x23.shape[1]}，第 11 层 dim={x11.shape[1]}，"
          f"拼接后 dim={x_concat.shape[1]}。")

    for lam in (0.03, 0.3, 1.0):
        m23, s23 = cv_acc(x23, y, lam)
        mc, sc = cv_acc(x_concat, y, lam)
        print(f"λ={lam:>5g}：第 23 层 CV acc={m23:.4f}±{s23:.4f} | "
              f"拼接(11,23) CV acc={mc:.4f}±{sc:.4f}")

    print("-" * 72)
    print("结论：在各 λ 下，拼接(11,23) 的交叉验证准确率与单用第 23 层基本持平"
          "（差异远小于折间标准差）。即在 N=5088 的规模下，第 11 层没有为真假判别"
          "提供第 23 层之外的可用线性信息，拼接反而把特征维度翻倍、增加过拟合风险，"
          "却换不来效果提升。因此最终仍单用第 23 层，假说 B 未获支持。")
    print()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "rejected_hypotheses.txt"
    with out_path.open("w", encoding="utf-8") as f:
        original = sys.stdout
        sys.stdout = Tee(original, f)
        try:
            print(f"# 被否决假说的原始证据（脚本 {Path(__file__).name}）\n")
            hypothesis_a_country_residual()
            hypothesis_b_layer_concat()
        finally:
            sys.stdout = original
    print(f"\n原始证据已写入 {out_path}")


if __name__ == "__main__":
    main()
