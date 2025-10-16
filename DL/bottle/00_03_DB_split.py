from pymongo import MongoClient
import gridfs
import os
import random
from PIL import Image

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["image_dataset"]
fs = gridfs.GridFS(db)

# 클래스 정의
class_names = [
    "bad-broken_large",
    "bad-broken_small",
    "bad-contamination",
    "bottle-good"
]
class_map = {name: idx for idx, name in enumerate(class_names)}

# split 비율
ratios = {"train": 0.8, "val": 0.15, "test": 0.05}

# 저장 경로
base_dir = "dataset/yolo"
splits = ["train", "val", "test"]

# 폴더 생성
for split in splits:
    os.makedirs(f"{base_dir}/{split}/images", exist_ok=True)
    os.makedirs(f"{base_dir}/{split}/labels", exist_ok=True)

# 클래스별 분할 및 저장
for cls in class_names:
    docs = list(fs.find({
        "metadata.class": cls,
        "metadata.source": {"$in": ["total", "augmented"]}
    }))
    random.shuffle(docs)

    total = len(docs)
    train_end = int(total * ratios["train"])
    val_end = train_end + int(total * ratios["val"])

    for i, file_doc in enumerate(docs):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        # 이미지 저장
        filename = file_doc.filename
        image_path = f"{base_dir}/{split}/images/{filename}"
        with open(image_path, "wb") as f:
            f.write(file_doc.read())

        # 이미지 크기 확인
        image = Image.open(image_path)
        w, h = image.size

        # 라벨 저장
        label_path = f"{base_dir}/{split}/labels/{filename.replace('.jpg', '.txt')}"
        with open(label_path, "w") as f:
            for label in file_doc.metadata["labels"]:
                cls_id = class_map[label["class"]]
                x = label["bbox"]["x"]
                y = label["bbox"]["y"]
                bw = label["bbox"]["width"]
                bh = label["bbox"]["height"]

                # YOLO 형식으로 변환
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")