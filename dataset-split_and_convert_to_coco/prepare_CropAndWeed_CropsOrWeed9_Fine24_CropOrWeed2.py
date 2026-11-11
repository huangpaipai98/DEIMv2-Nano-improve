#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fresh CropAndWeed -> COCO builder using the OFFICIAL CropAndWeed mapping.

Variants:
    CropsOrWeed9
    Fine24
    CropOrWeed2

SOURCE OF TRUTH:
    data_original/images/
    data_original/bboxes/
    data_original/labelIds/CropAndWeed/

OFFICIAL MAPPING:
    cropandweed-dataset/cnw/map_dataset.py
    cropandweed-dataset/cnw/utilities/datasets.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required: pip install pillow", file=sys.stderr)
    raise


SEED = 42
VARIANTS = ("CropsOrWeed9", "Fine24", "CropOrWeed2")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_original"),
        help="Original CropAndWeed data root",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("cropandweed-dataset"),
        help="Official cropandweed-dataset repository",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("CropAndWeed"),
        help="COCO output directory",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output-root and rebuild from data_original",
    )
    return p.parse_args()


def find_images(images_dir: Path) -> Dict[str, Path]:
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    result = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix in exts:
            if p.stem in result:
                raise RuntimeError(f"Duplicate image stem: {p.stem}")
            result[p.stem] = p
    return result


def find_csvs(bboxes_dir: Path) -> Dict[str, Path]:
    result = {}
    for p in bboxes_dir.glob("*.csv"):
        if p.is_file():
            if p.stem in result:
                raise RuntimeError(f"Duplicate bbox stem: {p.stem}")
            result[p.stem] = p
    return result


def read_csv_rows(path: Path):
    """
    The official map_dataset.py reads the source files using DictReader with
    these exact field names and no header requirement.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            f,
            fieldnames=[
                "left", "top", "right", "bottom",
                "label_id", "stem_x", "stem_y",
            ],
        )
        for row in reader:
            yield row


def make_split(stems: List[str], seed: int):
    stems = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = round(n * 0.70)
    n_val = round(n * 0.15)

    return {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }


def write_splits(output_root: Path, splits):
    d = output_root / "splits"
    d.mkdir(parents=True, exist_ok=True)

    for split, stems in splits.items():
        with (d / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(["image_stem"])
            w.writerows([[s] for s in stems])


def prepare_official_input(data_root: Path, work_root: Path):
    """
    Build the directory layout expected by the official map_dataset.py:

        work_root/
        ├── bboxes/CropAndWeed/*.csv
        └── labelIds/CropAndWeed/*.png

    The files themselves are symlinks back to data_original, so no source
    dataset is duplicated or modified.
    """
    source_bbox = data_root / "bboxes"
    source_masks = data_root / "labelIds" / "CropAndWeed"

    if not source_bbox.is_dir():
        raise FileNotFoundError(source_bbox)
    if not source_masks.is_dir():
        raise FileNotFoundError(source_masks)

    bbox_target = work_root / "bboxes" / "CropAndWeed"
    mask_target = work_root / "labelIds" / "CropAndWeed"
    bbox_target.mkdir(parents=True)
    mask_target.mkdir(parents=True)

    for p in source_bbox.glob("*.csv"):
        (bbox_target / p.name).symlink_to(p.resolve())

    for p in source_masks.iterdir():
        if p.is_file():
            (mask_target / p.name).symlink_to(p.resolve())


def run_official_map_dataset(repo_root: Path, work_root: Path, target: str):
    """
    ACTUALLY RUN the official cnw/map_dataset.py.

    We intentionally use subprocess rather than reimplementing its mapping.
    """
    map_script = repo_root / "cnw" / "map_dataset.py"

    if not map_script.exists():
        raise FileNotFoundError(
            f"Official map_dataset.py not found: {map_script}"
        )

    bboxes = work_root / "bboxes"
    labelids = work_root / "labelIds"

    cmd = [
        sys.executable,
        str(map_script),
        "--bboxes", str(bboxes),
        "--labelids", str(labelids),
        "--dataset_source", "CropAndWeed",
        "--dataset_target", target,
    ]

    print()
    print("=" * 72)
    print(f"OFFICIAL map_dataset.py -> {target}")
    print("=" * 72)
    print("Command:")
    print(" ".join(cmd))
    print()

    # map_dataset.py imports "utilities" relative to cnw/, so cwd must be cnw.
    subprocess.run(
        cmd,
        cwd=str(repo_root / "cnw"),
        check=True,
    )

    mapped_dir = work_root / "bboxes" / target
    eval_dir = work_root / "bboxes" / f"{target}Eval"

    if not mapped_dir.is_dir():
        raise RuntimeError(f"Official mapping did not create {mapped_dir}")
    if not eval_dir.is_dir():
        raise RuntimeError(f"Official mapping did not create {eval_dir}")

    return mapped_dir, eval_dir


def load_official_categories(repo_root: Path, target: str):
    """
    Read category names/IDs directly from the official datasets.py.

    No manual mapping table is used.
    """
    utilities_dir = repo_root / "cnw"
    sys.path.insert(0, str(utilities_dir))

    try:
        from utilities import datasets
    finally:
        # Keep sys.path clean after import.
        try:
            sys.path.remove(str(utilities_dir))
        except ValueError:
            pass

    ds = datasets.DATASETS[target]

    categories = []
    for official_id in ds.get_label_ids():
        categories.append(
            {
                "official_id": int(official_id),
                "name": ds.get_label_name(official_id),
            }
        )

    return categories


def validate_bbox(left, top, right, bottom, width, height, source_path, row_num):
    if right <= left or bottom <= top:
        raise RuntimeError(
            f"Invalid mapped bbox: {source_path}, row={row_num}, "
            f"({left}, {top}, {right}, {bottom})"
        )

    if left < 0 or top < 0 or right > width or bottom > height:
        raise RuntimeError(
            f"Mapped bbox outside image: {source_path}, row={row_num}, "
            f"bbox=({left},{top},{right},{bottom}), "
            f"image=({width},{height})"
        )


def build_variant(
    variant: str,
    mapped_bbox_dir: Path,
    splits,
    image_map,
    output_root,
    repo_root,
):
    official_categories = load_official_categories(repo_root, variant)

    # Official dataset IDs are zero-based for these predefined variants.
    # COCO IDs are made contiguous and 1-based.
    official_to_coco = {
        item["official_id"]: i + 1
        for i, item in enumerate(official_categories)
    }

    categories = [
        {
            "id": official_to_coco[item["official_id"]],
            "name": item["name"],
        }
        for item in official_categories
    ]

    ann_dir = output_root / "annotations" / variant
    ann_dir.mkdir(parents=True, exist_ok=True)

    with (ann_dir / "category_mapping.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "dataset": "CropAndWeed",
                "variant": variant,
                "mapping_source": "official cnw/utilities/datasets.py",
                "mapping_execution": "official cnw/map_dataset.py",
                "official_categories": official_categories,
                "coco_id_mapping": {
                    str(k): v for k, v in official_to_coco.items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary = {}

    for split in ("train", "val", "test"):
        coco = {
            "info": {
                "description": f"CropAndWeed - {variant}",
                "source": "CropAndWeed Dataset",
                "mapping": "official map_dataset.py",
                "split": split,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }

        next_ann_id = 1
        zero_box = 0
        source_label_counter = Counter()
        mapped_label_counter = Counter()

        for image_id, stem in enumerate(splits[split], start=1):
            image_path = image_map[stem]
            mapped_csv = mapped_bbox_dir / f"{image_path.stem}.csv"

            with Image.open(image_path) as im:
                width, height = im.size

            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": f"{image_path.name}",
                    "width": width,
                    "height": height,
                }
            )

            box_count = 0

            # Important:
            # Official map_dataset.py creates a CSV in the target directory
            # ONLY when at least one source instance maps to that target.
            # Therefore an absent target CSV means zero mapped boxes.
            if mapped_csv.exists():
                for row_num, row in enumerate(
                    read_csv_rows(mapped_csv), start=1
                ):
                    left = float(row["left"])
                    top = float(row["top"])
                    right = float(row["right"])
                    bottom = float(row["bottom"])
                    official_id = int(row["label_id"])

                    if official_id not in official_to_coco:
                        raise RuntimeError(
                            f"Unexpected official target ID {official_id} "
                            f"in {mapped_csv}"
                        )

                    validate_bbox(
                        left, top, right, bottom,
                        width, height,
                        mapped_csv,
                        row_num,
                    )

                    coco_id = official_to_coco[official_id]

                    source_label_counter[official_id] += 1
                    mapped_label_counter[coco_id] += 1

                    coco["annotations"].append(
                        {
                            "id": next_ann_id,
                            "image_id": image_id,
                            "category_id": coco_id,
                            "bbox": [
                                left,
                                top,
                                right - left,
                                bottom - top,
                            ],
                            "area": (right - left) * (bottom - top),
                            "iscrowd": 0,
                        }
                    )

                    next_ann_id += 1
                    box_count += 1

            if box_count == 0:
                zero_box += 1

        out_json = ann_dir / f"instances_{split}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)

        summary[split] = {
            "images": len(coco["images"]),
            "boxes": len(coco["annotations"]),
            "zero_box_images": zero_box,
            "category_counts": {
                str(k): v
                for k, v in sorted(mapped_label_counter.items())
            },
        }

        print(
            f"{variant}/{split}: "
            f"images={len(coco['images']):,}, "
            f"boxes={len(coco['annotations']):,}, "
            f"zero-box={zero_box:,}"
        )

    with (ann_dir / "dataset_info.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "dataset": "CropAndWeed",
                "variant": variant,
                "num_classes": len(categories),
                "mapping_source": "official cnw/utilities/datasets.py",
                "mapping_execution": "official cnw/map_dataset.py",
                "split_seed": None,
                "splits": summary,
                "notes": [
                    "Images are shared across variants.",
                    "COCO category IDs are 1-based; official mapped IDs are preserved in category_mapping.json.",
                    "The training/evaluation bbox source is the official target directory, not the Eval directory.",
                    "The official Eval directory contains fallback label 255 for unmapped/tiny/ambiguous instances and is not used for training annotations.",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return summary


def main():
    args = parse_args()

    data_root = args.data_root.resolve()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()

    print("=== CropAndWeed official-mapping COCO builder ===")
    print(f"Original data: {data_root}")
    print(f"Official repo: {repo_root}")
    print(f"Output:        {output_root}")
    print(f"Seed:          {args.seed}")
    print(f"Variants:      {', '.join(VARIANTS)}")
    print()

    # Required source checks.
    for p in [
        data_root / "images",
        data_root / "bboxes",
        data_root / "labelIds" / "CropAndWeed",
        repo_root / "cnw" / "map_dataset.py",
        repo_root / "cnw" / "utilities" / "datasets.py",
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Required path missing: {p}")

    if output_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"{output_root} already exists. "
                "Use --overwrite for a guaranteed fresh build."
            )
        print(f"Removing existing output: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)

    image_map = find_images(data_root / "images")
    bbox_map = find_csvs(data_root / "bboxes")

    if set(image_map) != set(bbox_map):
        missing_bbox = sorted(set(image_map) - set(bbox_map))
        missing_image = sorted(set(bbox_map) - set(image_map))
        raise RuntimeError(
            f"Image/CSV mismatch. Missing bbox={missing_bbox[:10]}, "
            f"missing image={missing_image[:10]}"
        )

    stems = sorted(image_map)

    print(f"Images:    {len(image_map):,}")
    print(f"Bbox CSVs: {len(bbox_map):,}")
    print(f"Matched:   {len(stems):,}")

    splits = make_split(stems, args.seed)
    print(
        f"Fresh split: train={len(splits['train']):,}, "
        f"val={len(splits['val']):,}, "
        f"test={len(splits['test']):,}"
    )

    write_splits(output_root, splits)

    # Temporary directory is the only intermediate mapping workspace.
    # It contains symlinks to data_original and official mapped outputs.
    with tempfile.TemporaryDirectory(
        prefix="cropandweed_official_mapping_",
        dir=str(data_root.parent),
    ) as tmp:
        work_root = Path(tmp)
        prepare_official_input(data_root, work_root)

        mapped_dirs = {}

        for variant in VARIANTS:
            mapped_dir, eval_dir = run_official_map_dataset(
                repo_root,
                work_root,
                variant,
            )
            mapped_dirs[variant] = (mapped_dir, eval_dir)

        # Shared images are copied once only.
        for split in ("train", "val", "test"):
            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for stem in splits[split]:
                shutil.copy2(
                    image_map[stem],
                    split_dir / image_map[stem].name,
                )

        print()
        print("=" * 72)
        print("CONVERTING OFFICIAL MAPPED BBOXES TO COCO")
        print("=" * 72)

        all_summary = {}

        for variant in VARIANTS:
            mapped_dir, _eval_dir = mapped_dirs[variant]
            all_summary[variant] = build_variant(
                variant=variant,
                mapped_bbox_dir=mapped_dir,
                splits=splits,
                image_map=image_map,
                output_root=output_root,
                repo_root=repo_root,
            )

        # Update dataset_info with actual split seed after variant build.
        for variant in VARIANTS:
            info_path = (
                output_root
                / "annotations"
                / variant
                / "dataset_info.json"
            )
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["split_seed"] = args.seed
            info["split_method"] = "fresh deterministic shuffle of sorted image stems"
            info_path.write_text(
                json.dumps(info, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    for variant in VARIANTS:
        print(f"\n[{variant}]")
        for split in ("train", "val", "test"):
            c = all_summary[variant][split]
            print(
                f"  {split}: "
                f"images={c['images']:,}, "
                f"boxes={c['boxes']:,}, "
                f"zero-box={c['zero_box_images']:,}"
            )

    print("\nOutput:")
    print(output_root)
    print("├── train/")
    print("├── val/")
    print("├── test/")
    print("├── splits/")
    print("│   ├── train.csv")
    print("│   ├── val.csv")
    print("│   └── test.csv")
    print("└── annotations/")
    print("    ├── CropsOrWeed9/")
    print("    ├── Fine24/")
    print("    └── CropOrWeed2/")
    print("\nTemporary official mapping directory has been removed.")
    print("DONE.")


if __name__ == "__main__":
    main()
