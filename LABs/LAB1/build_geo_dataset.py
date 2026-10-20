"""构造 world 任务的扩充数据集（附加内容，文档 §2.4）。

数据来源：GeoNames（https://www.geonames.org/）公开数据集
- cities15000.txt：人口 >= 15000 的城市，含 名称/纬度/经度/ISO2 国家码
- countryInfo.txt：ISO2 国家码 -> 英文全名

目的：原训练集只有 500 条、158 个国家，留国家交叉验证显示"整国未见"时模型几乎
退化到全局均值（见 extra_analysis.py）。本脚本从 GeoNames 生成同模板句子扩充训练
集，并另建两份独立评估集：

- eval_seen：国家参与训练、但具体城市未参与训练，模拟"见过国家、未见地点"；
- eval_unseen：国家完全不出现在任何训练数据（原始 train/valid 与扩充集）中，
  模拟"未见国家"。

为避免"未见"名不副实，国家名先做归一化（小写、去前导 the、去标点）再比较；
扩充集只取原始数据覆盖过的国家，把 GeoNames 独有的国家整块留给 eval_unseen。

反泄漏：原始 train.csv 与 valid.csv 出现过的 (place, country) 组合一律剔除。

句子模板与助教原始数据一致：
    What's the longitude and latitude of {place}, {country}?
因为特征取"国名最后一个 token"的隐表征（见 extract_intrep.py）。

用法：
    python build_geo_dataset.py --geonames-dir /tmp/opencode
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "datasets"

MAX_CITIES_PER_COUNTRY = 0
MIN_CITIES_PER_COUNTRY = 3
N_EVAL_SEEN_PER_COUNTRY = 1
N_EVAL_UNSEEN = 300
SEED = 20261021
TEMPLATE = "What's the longitude and latitude of {place}, {country}?"


def norm_country(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def norm_place(text: str) -> str:
    return " ".join(text.lower().replace(",", " ").split())


def load_country_names(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            iso2, name = parts[0].strip(), parts[4].strip()
            if iso2 and name:
                mapping[iso2] = name
    return mapping


def load_cities(path: Path, iso_to_name: dict[str, str]) -> list[dict]:
    cities: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 19:
                continue
            name, lat, lon, iso2 = parts[1], parts[4], parts[5], parts[8]
            country = iso_to_name.get(iso2)
            if not country or not name:
                continue
            try:
                cities.append(
                    {
                        "place": name.strip(),
                        "country": country,
                        "latitude": float(lat),
                        "longitude": float(lon),
                    }
                )
            except ValueError:
                continue
    return cities


def original_records() -> tuple[set[tuple[str, str]], set[str]]:
    pairs: set[tuple[str, str]] = set()
    countries: set[str] = set()
    for split in ("train", "valid"):
        with (DATA / "world" / f"{split}.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                country = (row.get("country") or "").strip()
                if country:
                    countries.add(norm_country(country))
                sentence = row["sentence"].strip()
                prefix = "What's the longitude and latitude of "
                if not sentence.startswith(prefix):
                    continue
                body = sentence[len(prefix) :].rstrip("?")
                place = body
                if country and body.endswith(country):
                    place = body[: -len(country)].rstrip(", ").strip()
                pairs.add((norm_place(place), norm_country(country)))
    return pairs, countries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sentence", "country", "latitude", "longitude"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sentence": TEMPLATE.format(place=row["place"], country=row["country"]),
                    "country": row["country"],
                    "latitude": f"{row['latitude']:.5f}",
                    "longitude": f"{row['longitude']:.5f}",
                }
            )
    print(f"写入 {path}: {len(rows)} 条")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geonames-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DATA / "world_geo")
    args = parser.parse_args()

    iso_to_name = load_country_names(args.geonames_dir / "countryInfo.txt")
    cities = load_cities(args.geonames_dir / "cities15000.txt", iso_to_name)
    print(f"GeoNames 城市总数: {len(cities)}，覆盖国家: {len({c['country'] for c in cities})}")

    excluded, original_countries = original_records()
    before = len(cities)
    cities = [
        c
        for c in cities
        if (norm_place(c["place"]), norm_country(c["country"])) not in excluded
    ]
    print(f"剔除原始 train/valid 出现过的 (place,country): {before - len(cities)} 条")

    by_country: dict[str, list[dict]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for city in cities:
        key = (norm_place(city["place"]), norm_country(city["country"]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        by_country[city["country"]].append(city)
    print(f"按 (place,country) 去重后剩余城市: {len(seen_pairs)}")

    covered: list[str] = []
    novel: list[str] = []
    for country, pool in by_country.items():
        if len(pool) < MIN_CITIES_PER_COUNTRY:
            continue
        (covered if norm_country(country) in original_countries else novel).append(country)
    covered.sort()
    novel.sort()
    print(f"原始数据覆盖的国家（可用于扩充）: {len(covered)}")
    print(f"原始数据未覆盖的新国家（留作 eval_unseen）: {len(novel)}")
    cap = MAX_CITIES_PER_COUNTRY if MAX_CITIES_PER_COUNTRY > 0 else None
    print(f"每国城市上限: {'不限' if cap is None else cap}")

    rng = random.Random(SEED)

    eval_seen: list[dict] = []
    aug: list[dict] = []
    for country in covered:
        pool = by_country[country][:]
        rng.shuffle(pool)
        k_seen = min(N_EVAL_SEEN_PER_COUNTRY, max(1, len(pool) // 5))
        eval_seen.extend(pool[:k_seen])
        rest = pool[k_seen:]
        aug.extend(rest[:cap] if cap is not None else rest)
    rng.shuffle(aug)
    rng.shuffle(eval_seen)

    eval_unseen: list[dict] = []
    per_novel = max(1, N_EVAL_UNSEEN // max(len(novel), 1))
    for country in novel:
        pool = by_country[country][:]
        rng.shuffle(pool)
        eval_unseen.extend(pool[:per_novel])
    rng.shuffle(eval_unseen)
    eval_unseen = eval_unseen[:N_EVAL_UNSEEN]

    print(f"扩充训练集: {len(aug)} 条，覆盖 {len(covered)} 国")
    print(f"eval_seen: {len(eval_seen)} 条；eval_unseen: {len(eval_unseen)} 条，"
          f"覆盖 {len(novel)} 个全新国家")

    write_csv(args.out_dir / "aug.csv", aug)
    write_csv(args.out_dir / "eval_seen.csv", eval_seen)
    write_csv(args.out_dir / "eval_unseen.csv", eval_unseen)
    with (args.out_dir / "meta.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"max_cities_per_country={MAX_CITIES_PER_COUNTRY}\n")
        stream.write(f"seed={SEED}\n")
        stream.write(f"covered_countries={len(covered)}\n")
        stream.write(f"novel_countries={len(novel)}\n")
        stream.write(f"aug_rows={len(aug)}\n")
        stream.write(f"eval_seen_rows={len(eval_seen)}\n")
        stream.write(f"eval_unseen_rows={len(eval_unseen)}\n")
    print(f"元信息写入 {args.out_dir / 'meta.txt'}")


if __name__ == "__main__":
    main()
