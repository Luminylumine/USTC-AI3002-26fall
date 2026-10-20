"""用 GeoNames 扩充集训练最终 world 探针，并在官方验证集与自建评估集上评估。

流程：
1. 训练集 = 原始 500 条 + GeoNames 扩充集（全部或前 K 条）；
2. 对纬度、经度分别扫 λ（与已提交模型的"逐输出调 λ"口径一致）；
3. 用闭式岭回归求解（同一目标函数的精确最优；文档允许提交权重不由 2.1 的手写
   模型训练得到，且思考题 3 已证明闭式解与手写 GD 数值一致）；
4. 在 original_valid / geo_eval_seen / geo_eval_unseen 三个集合上报告；
5. 写出可替换 results/world.npz 的权重到 --output。

用法：
    python train_final_geo.py --layer 11 --mode unweighted
    python train_final_geo.py --layer 11 --mode unweighted --size 5000
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geo_augment_experiment import (
    country_weights,
    evaluate,
    load_geo,
    precompute,
    solve,
    train_countries,
)
from train23 import load_split

HERE = Path(__file__).parent
DATA = HERE / "datasets"
RESULTS = HERE / "results"

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


def solve_output(pre, lam: float, index: int) -> tuple[np.ndarray, float]:
    """对单个输出维度求解岭回归（lat 或 lon 各自选 λ 时使用）。"""
    mean_x, mean_y, gram, cross = pre
    d = gram.shape[0]
    w = np.linalg.solve(gram + lam * np.eye(d), cross[:, index])
    b = float(mean_y[index] - mean_x @ w)
    return w, b


def build(layer: int, size: int, mode: str):
    x_base, y_base = load_split("world", "train", layer)
    x_aug, y_aug = load_geo("aug", layer)
    k = len(y_aug) if size <= 0 else min(size, len(y_aug))
    x = np.concatenate([x_base, x_aug[:k]], axis=0)
    y = np.concatenate([y_base, y_aug[:k]], axis=0)
    countries = train_countries()[: len(y_base) + k]
    weights = None if mode == "unweighted" else country_weights(countries)
    return x, y, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=11, choices=(11, 23, 35))
    parser.add_argument("--mode", choices=("unweighted", "weighted"), default="unweighted")
    parser.add_argument("--size", type=int, default=0, help="扩充集条数，0 表示全部")
    parser.add_argument("--output", type=Path, default=RESULTS / "world_geo.npz")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    original = sys.stdout
    sys.stdout = Tee(original, buffer)
    try:
        run(args)
    finally:
        sys.stdout = original
    (RESULTS / "world_geo_report.txt").write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\n报告已写入 {RESULTS / 'world_geo_report.txt'}")


def run(args) -> None:
    layer = args.layer
    x, y, weights = build(layer, args.size, args.mode)
    print(f"# GeoNames 扩充最终模型（layer {layer}, mode={args.mode}）")
    print(f"训练集: {x.shape[0]} 条（原始 500 + 扩充 {x.shape[0] - 500}）\n")

    pre = precompute(x, y, weights)
    valid = load_split("world", "valid", layer)
    seen = load_geo("eval_seen", layer)
    unseen = load_geo("eval_unseen", layer)
    sets = {"valid": valid, "eval_seen": seen, "eval_unseen": unseen}

    print("逐输出 λ 扫描（纬度按 valid 纬度 MSE 选，经度同理）：")
    chosen = {}
    for index, name in ((0, "lat"), (1, "lon")):
        print(f"\n[{name}]")
        best = None
        for lam in LAMBDA_GRID:
            w, b = solve_output(pre, lam, index)
            mse = {}
            for key, (xs, ys) in sets.items():
                pred = xs @ w + b
                mse[key] = float(np.mean((pred - ys[:, index]) ** 2))
            print(
                f"  λ={lam:<6g} valid_{name}={mse['valid']:8.3f} "
                f"seen={mse['eval_seen']:8.3f} unseen={mse['eval_unseen']:9.3f}"
            )
            if best is None or mse["valid"] < best[0]:
                best = (mse["valid"], lam)
        chosen[name] = best[1]
        print(f"  → valid 最优 λ_{name}={best[1]:g} (MSE={best[0]:.3f})")

    lam_lat, lam_lon = chosen["lat"], chosen["lon"]
    print(f"\n选定: λ_lat={lam_lat:g}, λ_lon={lam_lon:g}")

    w_lat, b_lat = solve_output(pre, lam_lat, 0)
    w_lon, b_lon = solve_output(pre, lam_lon, 1)
    weights_out = np.asarray([w_lat, w_lon], dtype=np.float64)
    bias_out = np.asarray([b_lat, b_lon], dtype=np.float64)

    print("\n最终模型在三个集合上的表现：")
    print(f"{'集合':<16} {'样本数':>6} {'纬度MSE':>10} {'经度MSE':>10} "
          f"{'坐标平均MSE':>12} {'纬+经':>10}")
    for key, (xs, ys) in sets.items():
        pred = xs @ weights_out.T + bias_out
        lat = float(np.mean((pred[:, 0] - ys[:, 0]) ** 2))
        lon = float(np.mean((pred[:, 1] - ys[:, 1]) ** 2))
        print(
            f"{key:<16} {len(ys):>6} {lat:>10.3f} {lon:>10.3f} "
            f"{(lat + lon) / 2:>12.3f} {lat + lon:>10.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, weights=weights_out, bias=bias_out)
    print(f"\n权重已写入 {args.output} "
          f"(weights {weights_out.shape}, bias {bias_out.shape}, float64)")

    meta = {
        "layer": layer,
        "mode": args.mode,
        "n_train": int(x.shape[0]),
        "lambda_lat": lam_lat,
        "lambda_lon": lam_lon,
    }
    pd.Series(meta).to_json(RESULTS / "world_geo_meta.json", indent=2, force_ascii=False)
    print(f"元信息: {meta}")


if __name__ == "__main__":
    main()
