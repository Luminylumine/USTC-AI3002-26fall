"""Train the coordinate or truth probe and save its parameters."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model import LinearRegressionModel, LogisticRegressionModel


# 加载 task 任务的 train/valid 数据集，特征取自 layer 层
def load_split(task: str, split: str, layer: int) -> tuple[np.ndarray, np.ndarray]:

    # 在 Python 中，Path(__file__) 表示当前脚本文件的路径
    data_dir = Path(__file__).parent / "datasets" / task

    features = np.load(data_dir / f"{split}.layer{layer}.npy", allow_pickle=False)
    labels = pd.read_csv(data_dir / f"{split}.csv")
    if task == "world":
        targets = labels[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        # 形状为 (N, 2)
    else:
        targets = labels["label"].to_numpy(dtype=np.float64)
        # 形状为 (N,)
    if len(features) != len(targets):
        raise ValueError("The feature and label files have different row counts")

    return features.astype(np.float64), targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("world", "truth"))
    parser.add_argument("layer", type=int)
    parser.add_argument("output", type=Path, help="Path to the output .npz file")
    # 模型超参数（默认值即模型类中的默认值）
    parser.add_argument("--lambda1", type=float, default=0.0, help="L1 正则化强度")
    parser.add_argument("--lambda2", type=float, default=0.0, help="L2 正则化强度")
    parser.add_argument("--lambda2-lat", type=float, default=None, help="纬度 L2 正则化强度（缺省用 lambda2）")
    parser.add_argument("--lambda2-lon", type=float, default=None, help="经度 L2 正则化强度（缺省用 lambda2）")
    parser.add_argument("--weight-by-country", action="store_true", help="对 world 任务按 wi=1/sqrt(n_country) 加权")
    parser.add_argument("--learning-rate", type=float, default=None, help="固定学习率，缺省时自动估计")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="自动学习率的缩放系数")
    parser.add_argument("--lr-decay", type=float, default=0.0, help="学习率衰减系数")
    parser.add_argument("--momentum", type=float, default=0.0, help="动量系数")
    parser.add_argument("--max-iter", type=int, default=20_000, help="最大迭代次数")
    parser.add_argument("--tol", type=float, default=1e-10, help="损失相对改进阈值")
    parser.add_argument("--patience", type=int, default=50, help="容忍的停滞检查次数")
    parser.add_argument("--verbose", action="store_true", help="打印训练过程")
    args = parser.parse_args()

    model_kwargs = dict(
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        learning_rate=args.learning_rate,
        lr_scale=args.lr_scale,
        lr_decay=args.lr_decay,
        momentum=args.momentum,
        max_iter=args.max_iter,
        tol=args.tol,
        patience=args.patience,
        verbose=args.verbose,
    )

    # 加载数据
    train_x, train_y = load_split(args.task, "train", args.layer)
    valid_x, valid_y = load_split(args.task, "valid", args.layer)

    if args.task == "world":
        sample_weights = None
        if args.weight_by_country:
            labels_df = pd.read_csv(Path(__file__).parent / "datasets" / "world" / "train.csv")
            counts = labels_df["country"].value_counts()
            sample_weights = labels_df["country"].map(lambda c: 1.0 / np.sqrt(counts[c])).to_numpy(dtype=np.float64)
            print("Using sample weights: wi = 1 / sqrt(n_country)")

        # 纬度和经度各训练一个一维线性回归模型。
        latitude_targets = train_y[:, 0]
        longitude_targets = train_y[:, 1]

        lat_kwargs = dict(model_kwargs)
        lon_kwargs = dict(model_kwargs)
        if args.lambda2_lat is not None:
            lat_kwargs["lambda2"] = args.lambda2_lat
        if args.lambda2_lon is not None:
            lon_kwargs["lambda2"] = args.lambda2_lon

        latitude_model = LinearRegressionModel(train_x.shape[1], **lat_kwargs)
        longitude_model = LinearRegressionModel(train_x.shape[1], **lon_kwargs)
        latitude_model.fit(train_x, latitude_targets, sample_weight=sample_weights)
        longitude_model.fit(train_x, longitude_targets, sample_weight=sample_weights)

        latitude_predictions = latitude_model.predict(valid_x)
        longitude_predictions = longitude_model.predict(valid_x)
        prediction = np.column_stack((latitude_predictions, longitude_predictions))
        weights = np.stack((latitude_model.weights, longitude_model.weights))
        bias = np.array((latitude_model.bias, longitude_model.bias))
        train_prediction = np.column_stack(
            (latitude_model.predict(train_x), longitude_model.predict(train_x))
        )
        print(
            "Train iterations: "
            f"lat {latitude_model.n_iter_} (loss {latitude_model.train_loss_:.6f}), "
            f"lon {longitude_model.n_iter_} (loss {longitude_model.train_loss_:.6f})"
        )
        print(f"Training MSE: {np.mean((train_prediction - train_y) ** 2):.4f}")
    else:
        model = LogisticRegressionModel(train_x.shape[1], **model_kwargs)
        model.fit(train_x, train_y)
        prediction = model.predict(valid_x)
        weights = model.weights
        bias = model.bias
        print(f"Train iterations: {model.n_iter_} (loss {model.train_loss_:.6f})")
        train_prediction = model.predict(train_x)
        print(f"Training accuracy: {np.mean((train_prediction >= 0.5) == train_y):.4f}")

    if prediction.shape != valid_y.shape:
        raise ValueError(f"Expected predictions with shape {valid_y.shape}")

    if args.task == "world":
        print(f"Validation MSE: {np.mean((prediction - valid_y) ** 2):.4f}")
    else:
        print(f"Validation accuracy: {np.mean((prediction >= 0.5) == valid_y):.4f}")

    # 创造输出目录（如果不存在的话）
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 输出文件的名称被 output 参数指定，文件格式为 .npz
    np.savez(
        args.output,
        weights=np.asarray(weights, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
    )
    print(f"Saved {args.output} (layer {args.layer})")


if __name__ == "__main__":
    main()
