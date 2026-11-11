#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO detection TXT -> COCO detection JSON.

Inputs:
  image_dir   : image folder
  label_dir   : YOLO TXT label folder
  classes_txt : one class name per line; line 1 = class ID 0

YOLO row:
  class_id x_center y_center width height

The four coordinates are normalized to [0, 1].

COCO:
  category_id = YOLO class_id + 1
  bbox = [x_min, y_min, width, height] in absolute pixels
  file_name = image basename only, e.g. "00000001.jpg"

Images without a matching TXT are retained with an empty annotations list.
Label TXT files without a matching image are ignored with a warning.

Usage:
  python yolo_to_coco.py IMAGE_DIR LABEL_DIR CLASSES_TXT -o instances.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert YOLO detection labels to COCO JSON."
    )
    p.add_argument("image_dir", type=Path, help="Folder containing images.")
    p.add_argument("label_dir", type=Path, help="Folder containing YOLO .txt labels.")
    p.add_argument("classes_txt", type=Path, help="Path to classes.txt.")
    p.add_argument(
        "-o", "--output", type=Path, default=Path("instances.json"),
        help="Output COCO JSON path. Default: ./instances.json"
    )
    p.add_argument(
        "--skip-invalid", action="store_true",
        help="Skip malformed YOLO rows instead of stopping."
    )
    return p.parse_args()


def read_classes(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"classes.txt not found: {path}")

    classes = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)

    if not classes:
        raise RuntimeError(f"No class names found in {path}")
    return classes


def find_images(path: Path) -> Dict[str, Path]:
    if not path.is_dir():
        raise NotADirectoryError(f"Image directory not found: {path}")

    images: Dict[str, Path] = {}
    for p in sorted(path.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file() or p.suffix not in IMAGE_EXTENSIONS:
            continue
        if p.stem in images:
            raise RuntimeError(
                f"Duplicate image stem: {p.stem}\n"
                f"  {images[p.stem]}\n  {p}"
            )
        images[p.stem] = p

    if not images:
        raise RuntimeError(f"No images found in {path}")
    return images


def find_labels(path: Path) -> Dict[str, Path]:
    if not path.is_dir():
        raise NotADirectoryError(f"Label directory not found: {path}")

    labels: Dict[str, Path] = {}
    for p in sorted(path.glob("*.txt"), key=lambda x: x.name.lower()):
        if p.stem in labels:
            raise RuntimeError(f"Duplicate label stem: {p.stem}")
        labels[p.stem] = p
    return labels


def parse_yolo_row(
    line: str,
    label_path: Path,
    line_number: int,
    num_classes: int,
) -> Tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(
            f"{label_path}: line {line_number}: expected 5 values, got {len(parts)}\n"
            f"  {line}"
        )

    try:
        class_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
    except ValueError as exc:
        raise ValueError(
            f"{label_path}: line {line_number}: invalid numeric value\n  {line}"
        ) from exc

    if not 0 <= class_id < num_classes:
        raise ValueError(
            f"{label_path}: line {line_number}: class_id={class_id} "
            f"outside [0, {num_classes - 1}]"
        )

    eps = 1e-9
    for name, value in (
        ("x_center", xc), ("y_center", yc), ("width", w), ("height", h)
    ):
        if value < -eps or value > 1.0 + eps:
            raise ValueError(
                f"{label_path}: line {line_number}: {name}={value} "
                "outside normalized range [0, 1]"
            )

    if w <= 0 or h <= 0:
        raise ValueError(
            f"{label_path}: line {line_number}: width/height must be > 0, got {w}, {h}"
        )

    return class_id, xc, yc, w, h


def yolo_to_coco_bbox(
    xc: float, yc: float, w: float, h: float,
    image_width: int, image_height: int,
) -> List[float]:
    return [
        (xc - w / 2.0) * image_width,
        (yc - h / 2.0) * image_height,
        w * image_width,
        h * image_height,
    ]


def validate_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int,
    label_path: Path,
    line_number: int,
) -> None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(
            f"{label_path}: line {line_number}: invalid converted bbox {bbox}"
        )

    eps = 1e-6
    if (
        x < -eps
        or y < -eps
        or x + w > image_width + eps
        or y + h > image_height + eps
    ):
        raise ValueError(
            f"{label_path}: line {line_number}: converted bbox outside image\n"
            f"  bbox={bbox}\n"
            f"  image_size=({image_width}, {image_height})"
        )


def convert(
    image_dir: Path,
    label_dir: Path,
    classes_txt: Path,
    output: Path,
    skip_invalid: bool = False,
) -> None:
    classes = read_classes(classes_txt)
    images = find_images(image_dir)
    labels = find_labels(label_dir)

    extra_labels = sorted(set(labels) - set(images))
    if extra_labels:
        print(
            f"WARNING: {len(extra_labels)} label files have no matching image; ignored.",
            file=sys.stderr,
        )
        print("  Examples:", ", ".join(extra_labels[:10]), file=sys.stderr)

    coco = {
        "info": {
            "description": "YOLO to COCO conversion",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": i + 1,
                "name": name,
                "supercategory": "none",
            }
            for i, name in enumerate(classes)
        ],
    }

    annotation_id = 1
    invalid_rows = 0
    images_without_labels = 0

    for image_id, image_path in enumerate(
        sorted(images.values(), key=lambda p: p.name.lower()),
        start=1,
    ):
        with Image.open(image_path) as im:
            width, height = im.size

        # Only the filename is stored, e.g. "00000001.jpg".
        coco["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        label_path = labels.get(image_path.stem)
        if label_path is None:
            images_without_labels += 1
            continue

        with label_path.open("r", encoding="utf-8-sig") as f:
            for line_number, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue

                try:
                    class_id, xc, yc, w, h = parse_yolo_row(
                        line, label_path, line_number, len(classes)
                    )

                    bbox = yolo_to_coco_bbox(
                        xc, yc, w, h, width, height
                    )

                    validate_bbox(
                        bbox, width, height, label_path, line_number
                    )

                except ValueError as exc:
                    if skip_invalid:
                        invalid_rows += 1
                        print(
                            f"WARNING: skipped: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    raise

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                })
                annotation_id += 1

        if image_id == 1 or image_id % 500 == 0 or image_id == len(images):
            print(
                f"Processed {image_id:,}/{len(images):,}",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print()
    print("=== Conversion complete ===")
    print(f"Images:               {len(coco['images']):,}")
    print(f"Annotations:          {len(coco['annotations']):,}")
    print(f"Categories:            {len(coco['categories']):,}")
    print(f"Images without label:  {images_without_labels:,}")
    print(f"Invalid rows skipped:  {invalid_rows:,}")
    print(f"Output:                {output.resolve()}")


def main() -> None:
    args = parse_args()

    print("=== YOLO -> COCO ===")
    print(f"Images:  {args.image_dir.resolve()}")
    print(f"Labels:  {args.label_dir.resolve()}")
    print(f"Classes: {args.classes_txt.resolve()}")
    print(f"Output:  {args.output.resolve()}")
    print()

    convert(
        args.image_dir.resolve(),
        args.label_dir.resolve(),
        args.classes_txt.resolve(),
        args.output.resolve(),
        args.skip_invalid,
    )


if __name__ == "__main__":
    main()
