import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 解决 OpenMP 冲突

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import TQDM, colorstr

class COCODataset(YOLODataset):
    """Dataset that reads COCO JSON annotations directly without conversion to .txt files."""

    def __init__(self, *args, json_file="", **kwargs):
        """Initialize the dataset with a COCO JSON annotation file."""
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """Image paths are resolved from the JSON file, not from scanning a directory."""
        self.fraction = 1.0  # fraction is applied while scanning a directory, which this dataset skips
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """Parse COCO JSON and convert annotations to YOLO format. Results are saved to a .cache file."""
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        categories = {cat["id"]: i for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if not x["labels"]:
            raise RuntimeError(f"No images listed in {self.json_file} were found in {self.img_path}")
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Load labels from .cache file if available, otherwise parse JSON and create the cache."""
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]

class COCOTrainer(DetectionTrainer):
    """Trainer that uses COCODataset for direct COCO JSON training."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a COCODataset for the given split using the JSON file from the data config."""
        json_file = self.data["train_json"] if mode == "train" else self.data["val_json"]
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride = int(getattr(self.model, 'module', self.model).stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

class COCOValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        stride = self.stride.max() if hasattr(self.stride, "max") else self.stride

        return COCODataset(
            img_path=img_path,
            json_file=self.data["val_json"],
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(stride),
            pad=0.5,
            prefix=colorstr("val: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=1.0,
        )

if __name__ == '__main__':
    model = YOLO("yolo26n.yaml")
    model.train(
        data="yolo.yaml", 
        trainer=COCOTrainer, 
        epochs=300,
        batch=16,
        patience=50
    )
    best_pt = Path(model.trainer.best)
    model = YOLO(str(best_pt))
    model.val(
        data="yolo.yaml",
        validator=COCOValidator,
        batch=32,
        save_json=True,
    )