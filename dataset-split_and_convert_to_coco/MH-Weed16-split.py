from pathlib import Path
import random
import shutil


# ============================================================
# 配置
# ============================================================

# 原始图片文件夹
IMAGE_DIR = Path(r"images")

# 原始 YOLO 标签文件夹
LABEL_DIR = Path(r"labels")

# 划分后的数据集输出文件夹
OUTPUT_DIR = Path(r"MH-Weed16")

# 随机种子
SEED = 42

# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


# ============================================================
# 支持的图片格式
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# 创建输出目录
# ============================================================

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# 获取图片
# ============================================================

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

if len(image_files) == 0:
    raise RuntimeError(f"图片文件夹中没有找到图片：{IMAGE_DIR}")


# ============================================================
# 检查 YOLO 标签
# ============================================================

valid_pairs = []
missing_labels = []

for image_path in image_files:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        valid_pairs.append((image_path, label_path))
    else:
        missing_labels.append(image_path.name)


print("=" * 60)
print(f"图片总数          : {len(image_files)}")
print(f"找到对应标签      : {len(valid_pairs)}")
print(f"缺少标签的图片    : {len(missing_labels)}")
print("=" * 60)


# ============================================================
# 如果有图片没有对应标签，这里选择直接跳过
# ============================================================

if missing_labels:
    print("\n以下图片没有对应的 YOLO 标签，将不会参与划分：")
    for name in missing_labels[:20]:
        print("  ", name)

    if len(missing_labels) > 20:
        print(f"  ... 还有 {len(missing_labels) - 20} 个")


# ============================================================
# 固定随机种子并打乱
# ============================================================

random.seed(SEED)

random.shuffle(valid_pairs)


# ============================================================
# 按 TRAIN_RATIO : VAL_RATIO : TEST_RATIO 划分
# ============================================================

total = len(valid_pairs)

train_num = int(total * TRAIN_RATIO)
val_num = int(total * VAL_RATIO)
test_num = total - train_num - val_num

train_data = valid_pairs[:train_num]
val_data = valid_pairs[train_num:train_num + val_num]
test_data = valid_pairs[train_num + val_num:]


print("\n划分结果：")
print(f"Train : {len(train_data)}")
print(f"Val   : {len(val_data)}")
print(f"Test  : {len(test_data)}")
print(f"Total : {len(train_data) + len(val_data) + len(test_data)}")


# ============================================================
# 复制文件
# ============================================================

def copy_files(data, split_name):
    image_output = OUTPUT_DIR / "images" / split_name
    label_output = OUTPUT_DIR / "labels" / split_name

    for image_path, label_path in data:

        # 复制图片
        shutil.copy2(
            image_path,
            image_output / image_path.name
        )

        # 复制 YOLO 标签
        shutil.copy2(
            label_path,
            label_output / label_path.name
        )


copy_files(train_data, "train")
copy_files(val_data, "val")
copy_files(test_data, "test")


# ============================================================
# 输出最终目录
# ============================================================

print("\n数据集划分完成！")
print("\n输出目录：")
print(OUTPUT_DIR)

print("\n目录结构：")
print(f"""
{OUTPUT_DIR.name}/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
""")

print(f"比例：{TRAIN_RATIO*100} : {VAL_RATIO*100} : {TEST_RATIO*100}")
print("Random Seed:", SEED)