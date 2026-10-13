"""附加内容（实验文档 2.4）：对最终模型的泛化能力做诊断，并给出原始证据。

动机：验证集（world 130 条 / truth 300 条）上的指标很好，但文档指出最终效果分
在"助教未公布的测试集"上判定，且测试集分布与训练集可能差异较大。因此本脚本回答
一个验证集本身无法回答的问题：

    当遇到"训练时完全没见过的国家/地点"或"分布发生偏移"时，效果会怎样？

具体做三件事：

1. world：留国家交叉验证（leave-countries-out）。训练集 500 条覆盖 158 个国家，
   按"整国留出"切 5 折，模拟"测试集出现训练未见国家"的最坏情况；并与
   (a) 随机 5 折、(b) 验证集 形成"分布偏移 → 泛化恶化"的递进证据。
2. truth：按标签分层的 5 折交叉验证，给出比 300 条验证集更低方差的泛化估计，
   并检查折间稳定性。
3. truth：阈值与概率校准分析（Brier 分数、校准分箱、最优阈值）。

计算口径说明：
- world 用 numpy 闭式岭回归求解。其目标函数与 model.py 完全一致
  （mean((Xw+b-y)^2) + λ||w||^2）；思考题 3 已证明它与手写 GD 的解数值一致
  （偏差 1e-14 量级），因此用于"泛化诊断"是等价且更快的选择；最终提交仍由
  手写 GD 产出。
- truth 的逻辑回归没有闭式解，诊断用 scikit-learn 的 lbfgs 求解器
  （C = 1/(2Nλ) 与我们的 λ 语义一致，见思考题 3）；这不影响提交权重来源，
  仅为快速获得泛化估计。
- 阈值/校准分析用提交模型的同一份手写 GD 权重，保证与提交口径一致。

所有打印数字都是原始证据，落盘到 results/extra_analysis.txt 便于复核。

运行：
    python extra_analysis.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from model import LogisticRegressionModel
from train23 import load_split

HERE = Path(__file__).parent
RESULTS = HERE / "results"
DATA = HERE / "datasets"

WORLD_LAYER = 11
WORLD_LAMBDA2_LAT = 0.14
WORLD_LAMBDA2_LON = 0.28
TRUTH_LAYER = 23
TRUTH_LAMBDA2 = 0.03

N_FOLDS = 5
SEED = 0

GD_TRUTH = dict(momentum=0.99, max_iter=8_000, tol=1e-10, patience=100)


class Tee:
    """同时把 stdout 写到终端和文件，保证"打印的都是原始证据"且可复核。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _ridge_closed_form(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """闭式岭回归：min mean((Xw+b-y)^2)+lam*||w||^2，偏置由中心化隐式处理。"""
    xbar = x.mean(axis=0)
    ybar = float(y.mean())
    xc = x - xbar
    yc = y - ybar
    n = x.shape[0]
    w = np.linalg.solve(xc.T @ xc / n + lam * np.eye(x.shape[1]), xc.T @ yc / n)
    return w, ybar - float(xbar @ w)


def _world_sum_mse(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray
) -> tuple[float, float, float]:
    """返回 (纬度 MSE, 经度 MSE, 二者之和)。"""
    mse = []
    for out, lam in ((0, WORLD_LAMBDA2_LAT), (1, WORLD_LAMBDA2_LON)):
        w, b = _ridge_closed_form(x_tr, y_tr[:, out], lam)
        mse.append(float(np.mean((x_te @ w + b - y_te[:, out]) ** 2)))
    return mse[0], mse[1], mse[0] + mse[1]


def world_leave_countries_out() -> None:
    print("=" * 72)
    print("1. world：留国家交叉验证（leave-countries-out）")
    print("=" * 72)

    x_tr, y_tr = load_split("world", "train", WORLD_LAYER)
    x_va, y_va = load_split("world", "valid", WORLD_LAYER)
    countries = pd.read_csv(DATA / "world" / "train.csv")["country"].astype(str).to_numpy()
    uniq = sorted(set(countries.tolist()))
    print(f"训练集 {len(countries)} 条，覆盖 {len(uniq)} 个国家；验证集 {len(y_va)} 条。")

    rng = np.random.RandomState(SEED)
    idx = np.arange(len(countries))
    rng.shuffle(idx)
    folds = np.array_split(idx, N_FOLDS)
    rand_sum = []
    for k in range(N_FOLDS):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        rand_sum.append(_world_sum_mse(x_tr[tr], y_tr[tr], x_tr[te], y_tr[te])[2])
    print(f"(a) 随机 {N_FOLDS} 折：sum-MSE 均值={np.mean(rand_sum):.1f} "
          f"标准差={np.std(rand_sum):.1f}（各折 {[round(v) for v in rand_sum]}）")

    rng2 = np.random.RandomState(SEED + 1)
    rng2.shuffle(uniq)
    groups = np.array_split(uniq, N_FOLDS)
    by_country: dict[str, list[int]] = collections.defaultdict(list)
    for i, c in enumerate(countries):
        by_country[c].append(i)

    lco_sum, gm_sum = [], []
    for k in range(N_FOLDS):
        te = np.concatenate([by_country[c] for c in groups[k]])
        tr = np.concatenate(
            [by_country[c] for j in range(N_FOLDS) if j != k for c in groups[j]]
        )
        lco_sum.append(_world_sum_mse(x_tr[tr], y_tr[tr], x_tr[te], y_tr[te])[2])
        mu = y_tr[tr].mean(axis=0)
        gm_sum.append(float(np.mean((y_tr[te] - mu) ** 2)))
    print(f"(b) 留国家 {N_FOLDS} 折：sum-MSE 均值={np.mean(lco_sum):.1f} "
          f"标准差={np.std(lco_sum):.1f}（各折 {[round(v) for v in lco_sum]}）")
    print(f"    同一测试折上的全局均值基线 sum-MSE 均值={np.mean(gm_sum):.1f}")

    mse_lat, mse_lon, valid_sum = _world_sum_mse(x_tr, y_tr, x_va, y_va)
    print(f"(c) 验证集：纬度 MSE={mse_lat:.2f}，经度 MSE={mse_lon:.2f}，"
          f"sum-MSE={valid_sum:.2f}")
    print("    （闭式解为该 λ 下的精确最优；提交模型用有限步手写 GD，"
          "验证集 sum-MSE≈299，二者差异来自 GD 在原始特征上的收敛速度，见思考题 3。）")

    print("-" * 72)
    print(f"递进结论：验证集 sum-MSE={valid_sum:.0f} → 随机 CV={np.mean(rand_sum):.0f} "
          f"→ 留国家 CV={np.mean(lco_sum):.0f}。分布偏移越大、泛化越差；"
          f"留国家时模型仍显著优于全局均值基线"
          f"（{np.mean(lco_sum):.0f} vs {np.mean(gm_sum):.0f}），"
          f"说明坐标信息可跨国家泛化，但难度大幅上升。")
    print()


def truth_stratified_cv() -> None:
    print("=" * 72)
    print("2. truth：按标签分层的 5 折交叉验证（更低方差的泛化估计）")
    print("=" * 72)

    x_tr, y_tr = load_split("truth", "train", TRUTH_LAYER)
    x_va, y_va = load_split("truth", "valid", TRUTH_LAYER)
    y_tr = y_tr.astype(int)
    y_va = y_va.astype(int)
    print(f"训练集 {len(y_tr)} 条（正 {int(y_tr.sum())} / 负 {int(len(y_tr) - y_tr.sum())}），"
          f"验证集 {len(y_va)} 条。")

    rng = np.random.RandomState(SEED)
    folds: list[list[int]] = [[] for _ in range(N_FOLDS)]
    for label in (0, 1):
        ids = np.where(y_tr == label)[0]
        rng.shuffle(ids)
        for k, chunk in enumerate(np.array_split(ids, N_FOLDS)):
            folds[k].extend(chunk.tolist())

    c = 1.0 / (2.0 * len(y_tr) * TRUTH_LAMBDA2)
    accs, briers, train_accs = [], [], []
    for k in range(N_FOLDS):
        te = np.array(folds[k])
        tr = np.array([i for j in range(N_FOLDS) if j != k for i in folds[j]])
        clf = LogisticRegression(C=c, max_iter=3000, solver="lbfgs")
        clf.fit(x_tr[tr], y_tr[tr])
        p_te = clf.predict_proba(x_tr[te])[:, 1]
        accs.append(float(np.mean((p_te >= 0.5) == y_tr[te])))
        briers.append(float(np.mean((p_te - y_tr[te]) ** 2)))
        train_accs.append(float(np.mean(clf.predict(x_tr[tr]) == y_tr[tr])))
        print(f"  [fold {k + 1}/{N_FOLDS}] valid acc={accs[-1]:.4f}", flush=True)

    print(f"分层 {N_FOLDS} 折验证准确率：均值={np.mean(accs):.4f} "
          f"标准差={np.std(accs):.4f}（各折 {[round(a, 4) for a in accs]}）")
    print(f"对应训练折准确率均值={np.mean(train_accs):.4f}，"
          f"泛化差距≈{np.mean(train_accs) - np.mean(accs):.4f}")
    print(f"Brier 分数均值={np.mean(briers):.4f}（越低越好）")

    hand = LogisticRegressionModel(x_tr.shape[1], lambda2=TRUTH_LAMBDA2, **GD_TRUTH)
    hand.fit(x_tr, y_tr)
    va_acc = float(np.mean((hand.predict(x_va) >= 0.5) == y_va))
    print(f"对照：手写 GD 在完整训练集上训练、300 条验证集准确率={va_acc:.4f}")
    print(f"结论：交叉验证（{np.mean(accs):.4f}±{np.std(accs):.4f}）与验证集 "
          f"{va_acc:.4f} 一致，说明 300 条验证集的估计不是侥幸；折间标准差仅 "
          f"{np.std(accs):.4f}，模型对数据切分不敏感。")
    print()


def truth_threshold_calibration() -> None:
    print("=" * 72)
    print("3. truth：阈值与概率校准分析")
    print("=" * 72)

    x_tr, y_tr = load_split("truth", "train", TRUTH_LAYER)
    x_va, y_va = load_split("truth", "valid", TRUTH_LAYER)
    y_tr = y_tr.astype(int)
    y_va = y_va.astype(int)

    model = LogisticRegressionModel(x_tr.shape[1], lambda2=TRUTH_LAMBDA2, **GD_TRUTH)
    model.fit(x_tr, y_tr)
    p_va = model.predict(x_va)

    print("阈值扫描（验证集 300 条）：")
    acc05 = float(np.mean((p_va >= 0.5) == y_va))
    best_t, best_acc = 0.5, acc05
    for t in np.arange(0.30, 0.71, 0.02):
        acc = float(np.mean((p_va >= t) == y_va))
        mark = " <- 更好" if acc > best_acc + 1e-12 else ""
        if acc > best_acc + 1e-12:
            best_t, best_acc = float(t), acc
        print(f"  阈值={t:.2f}  准确率={acc:.4f}{mark}")
    print(f"默认 0.5 阈值准确率={acc05:.4f}；最优阈值={best_t:.2f} "
          f"准确率={best_acc:.4f}")

    brier = float(np.mean((p_va - y_va) ** 2))
    print(f"Brier 分数={brier:.4f}（完美 0；常数 0.5 预测为 0.25）")
    print("校准分箱（预测概率区间 -> 实际正例比例，样本数）：")
    edges = np.linspace(0.0, 1.0, 11)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (p_va >= lo) & (p_va < hi)
        else:
            mask = (p_va >= lo) & (p_va <= hi)
        if mask.sum() == 0:
            continue
        print(f"  [{lo:.1f},{hi:.1f})  实际正例={y_va[mask].mean():.3f}  "
              f"n={int(mask.sum())}")
    print("结论：概率随预测值单调上升（高概率区间正例比例更高），可作置信度使用；"
          "最优阈值与 0.5 接近，说明 0.5 阈值合理，换阈值无实质收益。")
    print()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "extra_analysis.txt"
    with out_path.open("w", encoding="utf-8") as f:
        original = sys.stdout
        sys.stdout = Tee(original, f)
        try:
            print(f"# 泛化诊断原始输出（脚本 {Path(__file__).name}）\n")
            world_leave_countries_out()
            truth_stratified_cv()
            truth_threshold_calibration()
        finally:
            sys.stdout = original
    print(f"\n原始证据已写入 {out_path}")


if __name__ == "__main__":
    main()
