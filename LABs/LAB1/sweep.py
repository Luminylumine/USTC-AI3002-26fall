"""实验 2.2 / 2.3 的超参数与层数网格搜索，并产出最终提交用的 .npz。

用法：
    python sweep.py            # 跑完整网格 + 训练最优配置，写出 results/ 下的 csv/json/npz
    python sweep.py --skip-final

说明：
- 所有"我们的模型"结果均来自 model.py 中手写的梯度下降（含 L1/L2 正则）；
- truth 任务额外用 sklearn 的逻辑回归做对照（仅作对照，不改变最终提交权重的来源）；
- world 的纬度/经度是两个独立的单输出回归，按 (b) 的结论可分别选正则强度。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from model import LinearRegressionModel, LogisticRegressionModel
from train23 import load_split

HERE = Path(__file__).parent
RESULTS = HERE / "results"

LAYERS = (11, 23, 35)
WORLD_LAMBDA2 = (0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5, 1.0)
WORLD_LAMBDA1 = (0.0, 1e-3)  # L1 选项（默认关，用于展示 L1 也可训练）
TRUTH_LAMBDA2 = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
TRUTH_SKLEARN_C = (1e-3, 1e-2, 1e-1, 1.0, 10.0)

GD_COMMON = dict(momentum=0.99, max_iter=20_000, tol=1e-10, patience=100)
TRUTH_GD_COMMON = dict(GD_COMMON, max_iter=8_000)


def _append(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # 所有行取键的并集，保证粗/细网格可以写进同一个 csv
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)



def sweep_world(verbose: bool = True) -> list[dict]:
    rows: list[dict] = []
    for layer in LAYERS:
        train_x, train_y = load_split("world", "train", layer)
        valid_x, valid_y = load_split("world", "valid", layer)
        for lam1 in WORLD_LAMBDA1:
            for lam2 in WORLD_LAMBDA2:
                t0 = time.time()
                preds = []
                train_preds = []
                iters = []
                for out in range(2):
                    model = LinearRegressionModel(
                        train_x.shape[1], lambda1=lam1, lambda2=lam2, **GD_COMMON
                    )
                    model.fit(train_x, train_y[:, out])
                    preds.append(model.predict(valid_x))
                    train_preds.append(model.predict(train_x))
                    iters.append(model.n_iter_)
                pred = np.column_stack(preds)
                train_pred = np.column_stack(train_preds)
                row = dict(
                    task="world",
                    layer=layer,
                    method="gd",
                    stage="coarse",
                    lambda1=lam1,
                    lambda2=lam2,
                    train_mse=float(np.mean((train_pred - train_y) ** 2)),
                    valid_mse=float(np.mean((pred - valid_y) ** 2)),
                    valid_mse_lat=float(np.mean((pred[:, 0] - valid_y[:, 0]) ** 2)),
                    valid_mse_lon=float(np.mean((pred[:, 1] - valid_y[:, 1]) ** 2)),
                    train_metric="",
                    valid_metric="",
                    n_iter="/".join(map(str, iters)),
                    seconds=round(time.time() - t0, 2),
                )
                rows.append(row)
                if verbose:
                    print(
                        f"[world] layer={layer:<3d} l1={lam1:<7g} l2={lam2:<8g} "
                        f"trainMSE={row['train_mse']:10.4f} validMSE={row['valid_mse']:10.4f} "
                        f"({row['seconds']}s)"
                    )
    _append(rows, RESULTS / "sweep_world.csv")
    return rows


def sweep_truth(verbose: bool = True) -> list[dict]:
    rows: list[dict] = []
    for layer in LAYERS:
        train_x, train_y = load_split("truth", "train", layer)
        valid_x, valid_y = load_split("truth", "valid", layer)

        for lam2 in TRUTH_LAMBDA2:
            t0 = time.time()
            model = LogisticRegressionModel(
                train_x.shape[1], lambda2=lam2, **TRUTH_GD_COMMON
            )
            model.fit(train_x, train_y)
            pred = model.predict(valid_x)
            train_pred = model.predict(train_x)
            row = dict(
                task="truth",
                layer=layer,
                method="gd",
                stage="coarse",
                lambda1=0.0,
                lambda2=lam2,
                train_mse="",
                valid_mse="",
                valid_mse_lat="",
                valid_mse_lon="",
                train_metric=f"{np.mean((train_pred >= 0.5) == train_y):.6f}",
                valid_metric=f"{np.mean((pred >= 0.5) == valid_y):.6f}",
                n_iter=model.n_iter_,
                seconds=round(time.time() - t0, 2),
            )
            rows.append(row)
            if verbose:
                print(
                    f"[truth] layer={layer:<3d} l2={lam2:<8g} "
                    f"trainAcc={row['train_metric']} validAcc={row['valid_metric']} "
                    f"({row['seconds']}s)"
                )

        # sklearn 对照（用于思考题 3）
        from sklearn.linear_model import LogisticRegression

        for C in TRUTH_SKLEARN_C:
            t0 = time.time()
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            clf.fit(train_x, train_y)
            row = dict(
                task="truth",
                layer=layer,
                method=f"sklearn(C={C:g})",
                stage="coarse",
                lambda1=0.0,
                lambda2="",
                train_mse="",
                valid_mse="",
                valid_mse_lat="",
                valid_mse_lon="",
                train_metric=f"{np.mean(clf.predict(train_x) == train_y):.6f}",
                valid_metric=f"{np.mean(clf.predict(valid_x) == valid_y):.6f}",
                n_iter=int(clf.n_iter_.max()),
                seconds=round(time.time() - t0, 2),
            )
            rows.append(row)
            if verbose:
                print(
                    f"[truth] layer={layer:<3d} sklearn C={C:<8g} "
                    f"trainAcc={row['train_metric']} validAcc={row['valid_metric']} "
                    f"({row['seconds']}s)"
                )
    _append(rows, RESULTS / "sweep_truth.csv")
    return rows


FINE_FACTORS = (0.5, 0.7, 1.4, 2.0)
TRUTH_FINE_FACTORS = (0.5, 2.0, 3.0, 4.0, 5.0)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("stage", "coarse")
    return rows


def _has_stage(rows: list[dict], stage: str) -> bool:
    return any(r.get("stage") == stage for r in rows)


def refine_world(rows: list[dict], verbose: bool = True) -> list[dict]:
    """在粗网格每个 layer 的最优 lambda2 附近做细网格（只跑 lambda1=0）。"""
    if _has_stage(rows, "fine"):
        print("world 细网格已存在，跳过")
        return rows
    new_rows: list[dict] = []
    for layer in LAYERS:
        base = [
            r for r in rows
            if r["method"] == "gd" and int(r["layer"]) == layer and float(r["lambda1"]) == 0.0
        ]
        if not base:
            continue
        lam_lat = float(min(base, key=lambda r: float(r["valid_mse_lat"]))["lambda2"])
        lam_lon = float(min(base, key=lambda r: float(r["valid_mse_lon"]))["lambda2"])
        existing = {round(float(r["lambda2"]), 12) for r in base}
        candidates = sorted(
            {round(max(lam * f, 1e-6), 12) for lam in (lam_lat, lam_lon) for f in FINE_FACTORS}
            - existing
        )
        train_x, train_y = load_split("world", "train", layer)
        valid_x, valid_y = load_split("world", "valid", layer)
        for lam2 in candidates:
            t0 = time.time()
            preds, train_preds, iters = [], [], []
            for out in range(2):
                model = LinearRegressionModel(
                    train_x.shape[1], lambda1=0.0, lambda2=lam2, **GD_COMMON
                )
                model.fit(train_x, train_y[:, out])
                preds.append(model.predict(valid_x))
                train_preds.append(model.predict(train_x))
                iters.append(model.n_iter_)
            pred = np.column_stack(preds)
            train_pred = np.column_stack(train_preds)
            row = dict(
                task="world",
                layer=layer,
                method="gd",
                stage="fine",
                lambda1=0.0,
                lambda2=lam2,
                train_mse=float(np.mean((train_pred - train_y) ** 2)),
                valid_mse=float(np.mean((pred - valid_y) ** 2)),
                valid_mse_lat=float(np.mean((pred[:, 0] - valid_y[:, 0]) ** 2)),
                valid_mse_lon=float(np.mean((pred[:, 1] - valid_y[:, 1]) ** 2)),
                train_metric="",
                valid_metric="",
                n_iter="/".join(map(str, iters)),
                seconds=round(time.time() - t0, 2),
            )
            new_rows.append(row)
            if verbose:
                print(
                    f"[world:fine] layer={layer:<3d} l2={lam2:<10g} "
                    f"validMSE={row['valid_mse']:9.4f} (lat {row['valid_mse_lat']:.4f} / "
                    f"lon {row['valid_mse_lon']:.4f}) ({row['seconds']}s)"
                )
    merged = list(rows) + new_rows
    _append(merged, RESULTS / "sweep_world.csv")
    return merged


def refine_truth(rows: list[dict], verbose: bool = True) -> list[dict]:
    if _has_stage(rows, "fine"):
        print("truth 细网格已存在，跳过")
        return rows
    new_rows: list[dict] = []
    for layer in LAYERS:
        base = [r for r in rows if r["method"] == "gd" and int(r["layer"]) == layer]
        if not base:
            continue
        lam0 = float(min(base, key=lambda r: -float(r["valid_metric"]))["lambda2"])
        if lam0 <= 0:
            lam0 = 1e-3
        existing = {round(float(r["lambda2"]), 12) for r in base}
        candidates = sorted(
            {round(lam0 * f, 12) for f in TRUTH_FINE_FACTORS} - existing
        )
        train_x, train_y = load_split("truth", "train", layer)
        valid_x, valid_y = load_split("truth", "valid", layer)
        for lam2 in candidates:
            t0 = time.time()
            model = LogisticRegressionModel(
                train_x.shape[1], lambda2=lam2, **TRUTH_GD_COMMON
            )
            model.fit(train_x, train_y)
            pred = model.predict(valid_x)
            train_pred = model.predict(train_x)
            row = dict(
                task="truth",
                layer=layer,
                method="gd",
                stage="fine",
                lambda1=0.0,
                lambda2=lam2,
                train_mse="",
                valid_mse="",
                valid_mse_lat="",
                valid_mse_lon="",
                train_metric=f"{np.mean((train_pred >= 0.5) == train_y):.6f}",
                valid_metric=f"{np.mean((pred >= 0.5) == valid_y):.6f}",
                n_iter=model.n_iter_,
                seconds=round(time.time() - t0, 2),
            )
            new_rows.append(row)
            if verbose:
                print(
                    f"[truth:fine] layer={layer:<3d} l2={lam2:<10g} "
                    f"trainAcc={row['train_metric']} validAcc={row['valid_metric']} "
                    f"({row['seconds']}s)"
                )
    merged = list(rows) + new_rows
    _append(merged, RESULTS / "sweep_truth.csv")
    return merged


def choose(world_rows: list[dict], truth_rows: list[dict]) -> dict:
    """按验证集指标选择：world 逐输出选 lambda2 再选层；truth 选 (layer, lambda2)。"""
    gd_world = [r for r in world_rows if r["method"] == "gd" and float(r["lambda1"]) == 0.0]

    world_choice: dict = {"per_output": []}
    best_total = np.inf
    for layer in LAYERS:
        rows = [r for r in gd_world if int(r["layer"]) == layer]
        lat = min(rows, key=lambda r: float(r["valid_mse_lat"]))
        lon = min(rows, key=lambda r: float(r["valid_mse_lon"]))
        total = float(lat["valid_mse_lat"]) + float(lon["valid_mse_lon"])
        world_choice["per_output"].append(
            {
                "layer": layer,
                "lambda2_lat": float(lat["lambda2"]),
                "lambda2_lon": float(lon["lambda2"]),
                "valid_mse_lat": float(lat["valid_mse_lat"]),
                "valid_mse_lon": float(lon["valid_mse_lon"]),
                "valid_mse_total": total,
            }
        )
        if total < best_total:
            best_total = total
            world_choice.update(
                layer=layer,
                lambda2_lat=float(lat["lambda2"]),
                lambda2_lon=float(lon["lambda2"]),
                valid_mse_total=total,
            )

    gd_truth = [r for r in truth_rows if r["method"] == "gd"]
    best_truth = min(gd_truth, key=lambda r: -float(r["valid_metric"]))
    truth_choice = dict(
        layer=int(best_truth["layer"]),
        lambda2=float(best_truth["lambda2"]),
        valid_accuracy=float(best_truth["valid_metric"]),
        train_accuracy=float(best_truth["train_metric"]),
    )

    return {"world": world_choice, "truth": truth_choice}


def train_final(choice: dict, verbose: bool = True) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    layer = choice["world"]["layer"]
    train_x, train_y = load_split("world", "train", layer)
    valid_x, valid_y = load_split("world", "valid", layer)
    weights, biases = [], []
    total_valid = 0.0
    for out, key in enumerate(("lat", "lon")):
        model = LinearRegressionModel(
            train_x.shape[1],
            lambda2=choice["world"][f"lambda2_{key}"],
            **GD_COMMON,
        )
        model.fit(train_x, train_y[:, out])
        pred = model.predict(valid_x)
        mse = float(np.mean((pred - valid_y[:, out]) ** 2))
        total_valid += mse
        weights.append(model.weights)
        biases.append(model.bias)
        if verbose:
            print(f"[world] {key}: lambda2={choice['world'][f'lambda2_{key}']} validMSE={mse:.4f}")
    np.savez(
        RESULTS / "world.npz",
        weights=np.asarray(weights, dtype=np.float64),
        bias=np.asarray(biases, dtype=np.float64),
    )
    if verbose:
        print(f"[world] layer={layer} total validMSE={total_valid:.4f} -> results/world.npz")

    layer = choice["truth"]["layer"]
    train_x, train_y = load_split("truth", "train", layer)
    valid_x, valid_y = load_split("truth", "valid", layer)
    model = LogisticRegressionModel(
        train_x.shape[1], lambda2=choice["truth"]["lambda2"], **TRUTH_GD_COMMON
    )
    model.fit(train_x, train_y)
    acc = float(np.mean((model.predict(valid_x) >= 0.5) == valid_y))
    np.savez(
        RESULTS / "truth.npz",
        weights=np.asarray(model.weights, dtype=np.float64),
        bias=np.asarray(model.bias, dtype=np.float64),
    )
    if verbose:
        print(f"[truth] layer={layer} lambda2={choice['truth']['lambda2']} "
              f"validAcc={acc:.4f} -> results/truth.npz")

    (RESULTS / "chosen.json").write_text(
        json.dumps(choice, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-final", action="store_true", help="只跑网格，不训练最终模型")
    parser.add_argument("--no-refine", action="store_true", help="不做最优 lambda 附近的细网格")
    parser.add_argument("--only", choices=("world", "truth"), help="只跑其中一个任务的网格")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    world_rows: list[dict] = []
    truth_rows: list[dict] = []
    if args.only in (None, "world"):
        if (RESULTS / "sweep_world.csv").exists():
            world_rows = _read_csv(RESULTS / "sweep_world.csv")
            print("复用已有的 results/sweep_world.csv")
        else:
            world_rows = sweep_world()
        if not args.no_refine:
            world_rows = refine_world(world_rows)
    if args.only in (None, "truth"):
        if (RESULTS / "sweep_truth.csv").exists():
            truth_rows = _read_csv(RESULTS / "sweep_truth.csv")
            print("复用已有的 results/sweep_truth.csv")
        else:
            truth_rows = sweep_truth()
        if not args.no_refine:
            truth_rows = refine_truth(truth_rows)

    if args.skip_final:
        return

    if world_rows and truth_rows:
        choice = choose(world_rows, truth_rows)
        print("\n=== 选择结果 ===")
        print(json.dumps(choice, indent=2, ensure_ascii=False))
        train_final(choice)


if __name__ == "__main__":
    main()
