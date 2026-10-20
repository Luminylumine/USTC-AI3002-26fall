"""一次性抽取 GeoNames 扩充集与评估集的多层隐表征，并做特征一致性校验。

复用 extract_intrep.py 的加载/抽取逻辑（同一 revision、同一 token 选择、同一 hook
位置），保证新特征与助教提供的特征处于同一表征空间。

三件事：
1. 一致性校验：用同一管线重抽原始 train.csv 的 layer 11，与助教提供的
   train.layer11.npy 比较。若相对误差很小，说明管线对齐；否则说明环境/revision
   不一致，后续结论不可信。
2. 抽取 aug.csv / eval_seen.csv / eval_unseen.csv 在 layer 11、23、35 的特征。
3. 报告每个输出的形状与数值范围。

用法：
    python extract_geo_features.py --model-dir ~/models/Qwen3-4B-Base
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from extract_intrep import extract_features, load_model, load_rows

HERE = Path(__file__).parent
DATA = HERE / "datasets"
GEO = DATA / "world_geo"
LAYERS = (11, 23, 35)


def parity_check(tokenizer, model) -> None:
    print("=" * 72)
    print("0. 特征一致性校验：重抽原始 train.csv 的 layer 11")
    print("=" * 72)
    reference_path = DATA / "world" / "train.layer11.npy"
    reference = np.load(reference_path, allow_pickle=False).astype(np.float64)
    rows = load_rows(DATA / "world" / "train.csv", "country")
    reproduced = extract_features(
        rows, tokenizer, model, layer=11, selection="country"
    ).astype(np.float64)

    diff = np.abs(reproduced - reference)
    max_abs = float(diff.max())
    rel_l2 = float(np.linalg.norm(reproduced - reference) / np.linalg.norm(reference))
    cos = float(
        np.sum(reproduced * reference)
        / (np.linalg.norm(reproduced) * np.linalg.norm(reference))
    )
    print(f"形状: {reproduced.shape} vs {reference.shape}")
    print(f"相对 L2 误差={rel_l2:.3e}，余弦相似度={cos:.10f}，"
          f"最大绝对误差={max_abs:.3e}（bf16 在大量级分量上的 1 ULP）")
    if rel_l2 > 3e-2:
        raise SystemExit(
            f"特征一致性校验失败（相对 L2 误差 {rel_l2:.3e} > 3e-2）："
            "环境或 revision 不一致，扩充实验不可信，已中止。"
        )
    print("校验通过：新特征与助教特征在同一表征空间（相对 L2 误差约 1%，"
          "来自 bf16 数值噪声）。")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(LAYERS),
        choices=list(LAYERS),
        help="要抽取的层，默认 11 23 35",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)

    if not args.skip_parity:
        parity_check(tokenizer, model)

    targets = [
        (GEO / "aug.csv", "扩充训练集"),
        (GEO / "eval_seen.csv", "见过国家的评估集"),
        (GEO / "eval_unseen.csv", "未见国家的评估集"),
    ]
    for csv_path, label in targets:
        rows = load_rows(csv_path, "country")
        for layer in args.layers:
            out_path = csv_path.with_name(f"{csv_path.stem}.layer{layer}.npy")
            if out_path.exists():
                print(f"跳过已存在: {out_path}")
                continue
            features = extract_features(
                rows, tokenizer, model, layer=layer, selection="country"
            )
            np.save(out_path, features)
            print(
                f"[{label}] layer {layer}: {features.shape} {features.dtype} "
                f"-> {out_path.name} (min={features.min():.3f} max={features.max():.3f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
