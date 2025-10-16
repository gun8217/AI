import cv2
from pathlib import Path
from ultralytics import YOLO
from pymongo import MongoClient

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["image_db"]
collection = db["images"]

# 설정
root = Path(__file__).parent.resolve()
model_path = Path(root / "runs/bottle3/weights/best.pt")
image_dir = Path(root / "dataset/images")
label_dir = Path(root / "dataset/labels")
crop_dir = Path(root / "dataset/crops")  # crop 저장 경로
crop_dir.mkdir(exist_ok=True)

# 클래스 이름 정의
class_map = {
    0: "bad-broken_large",
    1: "bad-broken_small",
    2: "bad-contamination",
    3: "bottle-good"
}

# 모델 로드
model = YOLO(model_path)

# 이미지 순회
for img_path in image_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # YOLO 탐지 (index 0만)
    results = model(str(img_path))
    boxes = results[0].boxes
    if boxes and len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_idx = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cls_name = class_map.get(cls_idx, f"class_{cls_idx}")
        filename = f"{img_path.stem}_0{img_path.suffix}"

        # crop 이미지 저장
        crop_img = img[y1:y2, x1:x2]
        crop_path = crop_dir / filename
        cv2.imwrite(str(crop_path), crop_img)

        # MongoDB 등록
        collection.insert_one({
            "filename": filename,
            "class": cls_name,
            "index": 0,
            "path": str(img_path),
            "source": "yolo",
            "corrected": True,
            "confidence": round(conf, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })
        print(f"YOLO 등록 + crop 저장: {filename} | class: {cls_name}")

    # Label 기반 crop
    label_path = label_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        print(f"라벨 없음: {label_path.name}")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, x_center, y_center, box_w, box_h = map(float, parts)
        cls_id = int(cls_id)
        cls_name = class_map.get(cls_id, f"class_{cls_id}")

        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        collection.insert_one({
            "filename": f"{img_path.stem}_{i}{img_path.suffix}",
            "class": cls_name,
            "index": i,
            "path": str(img_path),
            "source": "origin",
            "corrected": True,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })
        print(f"Origin 등록: {img_path.name} | class: {cls_name} | index: {i}")

print("YOLO + Label 기반 MongoDB 등록 + crop 저장 완료!")