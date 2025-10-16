from pymongo import MongoClient
from pathlib import Path

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["image_dataset"]
collection = db["images"]

# 루트 경로
root = Path(__file__).parent.resolve()

# 원본 디렉터리들
image_dirs = [
    root / 'dataset/origin/train/images',
    root / 'dataset/origin/valid/images'
]
label_dirs = [
    root / 'dataset/origin/train/labels',
    root / 'dataset/origin/valid/labels'
]

# 클래스 이름 매핑
class_names = {
    0: "bad-broken_large",
    1: "bad-broken_small",
    2: "bad-contamination",
    3: "bottle-good"
}

# # 등록 시작
# for img_dir, lbl_dir in zip(image_dirs, label_dirs):
#     for img_path in img_dir.glob("*.jpg"):
#         label_path = lbl_dir / (img_path.stem + ".txt")
#         if not label_path.exists():
#             continue  # 라벨 없으면 건너뜀

#         with open(label_path, "r") as f:
#             lines = f.readlines()

#         for idx, line in enumerate(lines):
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue  # 잘못된 라벨 형식

#             cls_id, x_center, y_center, width, height = map(float, parts)
#             cls_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")

#             # YOLO 좌표
#             doc = {
#                 "filename": img_path.name,
#                 "class": cls_name,
#                 "index": idx,
#                 "path": str(img_path),
#                 "source": "total",
#                 "corrected": False,
#                 "confidence": None,
#                 "bbox": {
#                     "x_center": x_center,
#                     "y_center": y_center,
#                     "width": width,
#                     "height": height
#                 }
#             }
#             collection.insert_one(doc)

count = collection.count_documents({"source": "total"})
print(f'"source": "total" 로 저장된 데이터 개수: {count}')