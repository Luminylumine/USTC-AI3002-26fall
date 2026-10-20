"""打包实验一提交文件。

要求规范：
提交格式：一个 .zip 格式文件，命名为“PB24000301_林思睿_LAB1.pdf”（同时备份一份 .zip 后缀防止系统校验扩展名）
内含以下文件及目录：
- code/：你的 LAB1 文件夹（去除掉 datasets 目录，排除缓存与临时文件）
- report.pdf：实验报告
- linear.11.npz：world 地理坐标线性回归模型权重（第 11 层）
- logistic.23.npz：truth 句子真假逻辑回归模型权重（第 23 层）
"""

import os
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).parent.resolve()
OUTPUT_DIR = HERE.parent  # z:\linsirui\AI3002\LABs

ZIP_STEM = "PB24000301_林思睿_LAB1"
ZIP_PDF_NAME = f"{ZIP_STEM}.pdf"
ZIP_ZIP_NAME = f"{ZIP_STEM}.zip"

# 源文件
SRC_REPORT_PDF = HERE / "report.pdf"
SRC_LINEAR_NPZ = HERE / "results" / "world.npz"
SRC_LOGISTIC_NPZ = HERE / "results" / "truth.npz"

assert SRC_REPORT_PDF.exists(), f"Missing {SRC_REPORT_PDF}"
assert SRC_LINEAR_NPZ.exists(), f"Missing {SRC_LINEAR_NPZ}"
assert SRC_LOGISTIC_NPZ.exists(), f"Missing {SRC_LOGISTIC_NPZ}"

# 临时打包目录
STAGING_DIR = HERE / "_staging"
if STAGING_DIR.exists():
    shutil.rmtree(STAGING_DIR)
STAGING_DIR.mkdir()

try:
    # 1. 复制根级文件
    shutil.copy2(SRC_REPORT_PDF, STAGING_DIR / "report.pdf")
    shutil.copy2(SRC_LINEAR_NPZ, STAGING_DIR / "linear.11.npz")
    shutil.copy2(SRC_LOGISTIC_NPZ, STAGING_DIR / "logistic.23.npz")

    # 2. 复制并整理 code/ 目录（排除 datasets, __pycache__, .git, report.pdf, 临时文件）
    code_dir = STAGING_DIR / "code"
    code_dir.mkdir()

    ignore_patterns = {
        "datasets",
        "__pycache__",
        ".git",
        "_staging",
        "report.pdf",
        "report.html",
        ZIP_PDF_NAME,
        ZIP_ZIP_NAME,
    }

    for item in HERE.iterdir():
        if item.name in ignore_patterns or item.name.startswith("."):
            continue
        dest = code_dir / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                dest,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
            )
        else:
            shutil.copy2(item, dest)

    print("Staging directory structure:")
    for root, dirs, files in os.walk(STAGING_DIR):
        rel = os.path.relpath(root, STAGING_DIR)
        level = rel.count(os.sep)
        indent = "  " * level
        base = os.path.basename(root)
        if rel != ".":
            print(f"{indent}[DIR] {base}")
        for f in files:
            print(f"{indent}  {f}")

    # 3. 创建压缩包
    for out_name in [ZIP_PDF_NAME, ZIP_ZIP_NAME]:
        out_path = OUTPUT_DIR / out_name
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(STAGING_DIR):
                for f in files:
                    file_path = Path(root) / f
                    arcname = file_path.relative_to(STAGING_DIR)
                    zf.write(file_path, arcname)
        print(f"Created: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

    # 4. 校验压缩包内容
    print("\nVerifying archive contents:")
    with zipfile.ZipFile(OUTPUT_DIR / ZIP_PDF_NAME, "r") as zf:
        infolist = zf.infolist()
        print(f"Total entries: {len(infolist)}")
        root_items = set()
        code_items = 0
        datasets_found = False
        for info in infolist:
            parts = info.filename.split("/")
            root_items.add(parts[0])
            if parts[0] == "code":
                code_items += 1
            if "datasets" in parts:
                datasets_found = True
        print("Root level items:", sorted(list(root_items)))
        print(f"Files in code/: {code_items}")
        print(f"Datasets folder included? {datasets_found} (MUST BE FALSE)")
        assert not datasets_found, "ERROR: datasets folder must not be included!"
        assert "report.pdf" in root_items, "Missing report.pdf in root"
        assert "linear.11.npz" in root_items, "Missing linear.11.npz in root"
        assert "logistic.23.npz" in root_items, "Missing logistic.23.npz in root"
        assert "code" in root_items, "Missing code directory in root"
        print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")

finally:
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)

