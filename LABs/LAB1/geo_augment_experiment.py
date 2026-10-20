"""附加内容（文档 §2.4）：用 GeoNames 扩充 world 训练集，检验泛化是否变好。

设计：
- 训练集 = 原始 500 条 + GeoNames 扩充的前 K 条（K = 0/1000/3000/全部）；
- 在三个集合上评估：
    (1) 原始 valid（130 条，与提交口径可比）；
    (2) eval_seen（140 条，国家参与训练、具体地点未见过）；
    (3) eval_unseen（274 条，国家完全未参与训练）；
- 三层表征（11/23/35）都跑，检验"数据变多后最优层是否会改变"。

计算口径：闭式岭回归（与 model.py 的损失一致，思考题 3 已证明其与手写 GD 等价），
对所有条件使用同一方法与同一 λ 网格，保证对照公平。最终提交权重仍由手写 GD 产出。

为了效率，先算 X_c^T X_c / N 与 X_c^T Y_c / N，再对不同 λ 只做一次线性求解。

用法：
    python geo_augment_experiment.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train23 import load_split

HERE = Path(__file__).parent
DATA = HERE / "datasets"
GEO = DATA / "world_geo"
RESULTS = HERE / "results"

LAYERS = (11, 23, 35)
LAMBDA_GRID = (0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.14, 0.2, 0.28, 0.5)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def load_geo(split: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(GEO / f"{split}.layer{layer}.npy", allow_pickle=False)
    table = pd.read_csv(GEO / f"{split}.csv")
    targets = table[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    if len(features) != len(targets):
        raise ValueError(f"{split} 特征与标签行数不一致")
    return features.astype(np.float64), targets


def precompute(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float64)
    total = sample_weight.sum()
    mean_x = (sample_weight[:, None] * x).sum(axis=0) / total
    mean_y = (sample_weight[:, None] * y).sum(axis=0) / total
    xc = x - mean_x
    yc = y - mean_y
    wx = xc * sample_weight[:, None]
    gram = wx.T @ xc / total
    cross = wx.T @ yc / total
    return mean_x, mean_y, gram, cross


def country_weights(countries: np.ndarray) -> np.ndarray:
    """wi = 1/sqrt(n_country)：平衡国家影响力，避免人口大国主导回归。"""
    counts = pd.Series(countries).value_counts()
    inverse = (1.0 / np.sqrt(counts)).to_dict()
    return np.array([inverse[c] for c in countries], dtype=np.float64)


def train_countries() -> np.ndarray:
    base = pd.read_csv(DATA / "world" / "train.csv")["country"].astype(str).to_numpy()
    aug = pd.read_csv(GEO / "aug.csv")["country"].astype(str).to_numpy()
    return np.concatenate([base, aug])


def solve(pre, lam: float) -> tuple[np.ndarray, np.ndarray]:
    mean_x, mean_y, gram, cross = pre
    d = gram.shape[0]
    w = np.linalg.solve(gram + lam * np.eye(d), cross)
    b = mean_y - mean_x @ w
    return w, b


def evaluate(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    pred = x @ w + b
    mse_lat = float(np.mean((pred[:, 0] - y[:, 0]) ** 2))
    mse_lon = float(np.mean((pred[:, 1] - y[:, 1]) ** 2))
    return mse_lat, mse_lon


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=list(LAYERS), choices=list(LAYERS)
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "" if list(args.layers) == list(LAYERS) else f"_L{'_'.join(map(str, args.layers))}"
    out_path = RESULTS / f"geo_augment{suffix}.txt"
    buffer = io.StringIO()
    original = sys.stdout
    sys.stdout = Tee(original, buffer)
    try:
        run(args.layers)
    finally:
        sys.stdout = original
    out_path.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\n原始证据已写入 {out_path}")


def run(layers: tuple[int, ...] = LAYERS) -> None:
    print("# GeoNames 扩充试验原始输出（脚本 geo_augment_experiment.py）\n")
    print(f"层: {layers}")

    valid = {layer: load_split("world", "valid", layer) for layer in layers}
    train = {layer: load_split("world", "train", layer) for layer in layers}
    aug = {layer: load_geo("aug", layer) for layer in layers}
    seen = {layer: load_geo("eval_seen", layer) for layer in layers}
    unseen = {layer: load_geo("eval_unseen", layer) for layer in layers}

    print("数据集规模：")
    print(f"  原始 train = {len(train[11][1])} 条；Geo aug = {len(aug[11][1])} 条")
    print(f"  eval_seen = {len(seen[11][1])} 条；eval_unseen = {len(unseen[11][1])} 条")
    print(f"  原始 valid = {len(valid[11][1])} 条")

    baselines = {}
    for name, store in (("valid", valid), ("eval_seen", seen), ("eval_unseen", unseen)):
        x, y = store[11]
        global_mean = train[11][1].mean(axis=0)
        baselines[name] = float(np.mean((global_mean - y) ** 2))
    print("\n全局均值基线 vector-sum-MSE：")
    for name, value in baselines.items():
        print(f"  {name}: {value:.1f}")
    print()

    summaries: list[dict] = []
    tr_countries = train_countries()
    for layer in layers:
        print("=" * 78)
        print(f"layer {layer}")
        print("=" * 78)
        x_tr_base, y_tr_base = train[layer]
        x_aug, y_aug = aug[layer]
        sizes = tuple(s for s in (0, 1000, 5000, 16000, len(y_aug)) if s <= len(y_aug))

        for size in sizes:
            if size == 0:
                x_train, y_train = x_tr_base, y_tr_base
            else:
                k = min(size, len(y_aug))
                x_train = np.concatenate([x_tr_base, x_aug[:k]], axis=0)
                y_train = np.concatenate([y_tr_base, y_aug[:k]], axis=0)
            countries = tr_countries[: len(y_tr_base) + max(size, 0)]
            weight_full = country_weights(countries)
            for mode in ("unweighted", "weighted"):
                weights = None if mode == "unweighted" else weight_full
                pre = precompute(x_train, y_train, weights)
                for lam in LAMBDA_GRID:
                    w, b = solve(pre, lam)
                    row = {
                        "layer": layer,
                        "size": size,
                        "n": len(y_train),
                        "mode": mode,
                        "lambda": lam,
                    }
                    for name, store in (
                        ("valid", valid),
                        ("eval_seen", seen),
                        ("eval_unseen", unseen),
                    ):
                        lat, lon = evaluate(*store[layer], w, b)
                        row[f"{name}_sum"] = lat + lon
                        row[f"{name}_avg"] = (lat + lon) / 2
                    summaries.append(row)
                    print(
                        f"  n={row['n']:>5d} {mode:<10} λ={lam:<5g} | "
                        f"valid sum={row['valid_sum']:8.1f} | "
                        f"seen sum={row['eval_seen_sum']:8.1f} | "
                        f"unseen sum={row['eval_unseen_sum']:8.1f}"
                    )
        print()

    table = pd.DataFrame(summaries)
    table.to_csv(RESULTS / "geo_augment.csv", index=False)
    print(f"完整结果表已写入 {RESULTS / 'geo_augment.csv'}\n")

    print("=" * 78)
    print("按 λ 取各集合最优时的汇总（用于观察数据量与泛化的关系）")
    print("=" * 78)
    for layer in layers:
        for mode in ("unweighted", "weighted"):
            print(f"\nlayer {layer} / {mode}")
            print(f"{'n':>6} | {'valid最优λ':>10} {'valid sum':>10} | "
                  f"{'seen最优λ':>9} {'seen sum':>9} | {'unseen最优λ':>11} {'unseen sum':>11}")
            sub_layer = table[(table.layer == layer) & (table["mode"] == mode)]
            for size in sorted(sub_layer["size"].unique()):
                sub = sub_layer[sub_layer["size"] == size]
                vi = sub.loc[sub.valid_sum.idxmin()]
                si = sub.loc[sub.eval_seen_sum.idxmin()]
                ui = sub.loc[sub.eval_unseen_sum.idxmin()]
                print(
                    f"{int(vi.n):>6} | {vi['lambda']:>10g} {vi.valid_sum:>10.1f} | "
                    f"{si['lambda']:>9g} {si.eval_seen_sum:>9.1f} | "
                    f"{ui['lambda']:>11g} {ui.eval_unseen_sum:>11.1f}"
                )


if __name__ == "__main__":
    main()
