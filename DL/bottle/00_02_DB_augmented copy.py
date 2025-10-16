from pymongo import MongoClient
from PIL import Image, ImageEnhance
from pathlib import Path
import gridfs
import io

client = MongoClient("mongodb://localhost:27017/")
db = client["image_dataset"]
collection = db["images"]
fs = gridfs.GridFS(db)

# 증강 대상 클래스
excluded_class = "bottle-good"
augmentations = ["flip_vertical", "flip_horizontal", "brightness", "saturation"]

# 증강 함수들
def flip_vertical(img, bbox):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    bbox["y_center"] = 1.0 - bbox["y_center"]
    return img, bbox

def flip_horizontal(img, bbox):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    bbox["x_center"] = 1.0 - bbox["x_center"]
    return img, bbox

def adjust_brightness(img, factor=1.5):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def adjust_saturation(img, factor=1.5):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

# 증강 시작
docs = list(collection.find({"source": "total", "class": {"$ne": excluded_class}}))

for doc in docs:
    img_path = Path(doc["path"])
    if not img_path.exists():
        continue

    img = Image.open(img_path).convert("RGB")
    bbox = doc["bbox"]
    base_name = img_path.stem
    ext = img_path.suffix

    for aug in augmentations:
        new_bbox = bbox.copy()
        aug_img = img.copy()

        if aug == "flip_vertical":
            aug_img, new_bbox = flip_vertical(aug_img, new_bbox)
        elif aug == "flip_horizontal":
            aug_img, new_bbox = flip_horizontal(aug_img, new_bbox)
        elif aug == "brightness":
            aug_img = adjust_brightness(aug_img)
        elif aug == "saturation":
            aug_img = adjust_saturation(aug_img)

        # 이미지 메모리 저장
        img_bytes = io.BytesIO()
        aug_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # GridFS에 저장
        file_id = fs.put(img_bytes, filename=f"{base_name}_{aug}{ext}")

        # MongoDB에 메타데이터 저장
        collection.insert_one({
            "file_id": file_id,
            "filename": f"{base_name}_{aug}{ext}",
            "class": doc["class"],
            "index": doc["index"],
            "source": "augmented",
            "aug_type": aug,
            "corrected": doc["corrected"],
            "confidence": doc["confidence"],
            "bbox": new_bbox
        })

# count_total = collection.count_documents({"source": "total"})
# count_augmented = collection.count_documents({"source": "augmented"})

# total_count = count_total + count_augmented
# print(f"총 등록된 데이터 개수: {total_count} (total: {count_total}, augmented: {count_augmented})")