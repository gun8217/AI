import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from pymongo import MongoClient

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["image_db"]
collection = db["images"]

# 경로 설정
root = Path(__file__).parent.resolve()
crop_dir = root / "dataset/crops"
output_base_dir = root / "dataset/final_split"
split_ratios = {'train': 0.8, 'val': 0.15, 'test': 0.05}

# 출력 디렉토리 생성
for split in split_ratios:
    (output_base_dir / split).mkdir(parents=True, exist_ok=True)

# 클래스별 파일 수집 (MongoDB 기준)
class_files = defaultdict(list)
docs = collection.find({"source": "yolo"})

for doc in docs:
    filename = doc["filename"]
    cls = doc["class"]
    file_path = crop_dir / filename
    if file_path.exists():
        class_files[cls].append(file_path)

# 클래스별로 셔플 후 비율에 따라 분할 + 이동
for cls, files in class_files.items():
    random.shuffle(files)
    total = len(files)
    train_end = int(total * split_ratios['train'])
    val_end = train_end + int(total * split_ratios['val'])

    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split, split_files in splits.items():
        split_class_dir = output_base_dir / split / cls
        split_class_dir.mkdir(parents=True, exist_ok=True)
        for file_path in split_files:
            dst_path = split_class_dir / file_path.name
            shutil.move(file_path, dst_path)

# crops 폴더 삭제 (비었을 경우)
if crop_dir.exists() and not any(crop_dir.iterdir()):
    crop_dir.rmdir()
    print("🧹 crops 폴더가 비어 있어 삭제했습니다!")
else:
    print("⚠️ crops 폴더가 비어 있지 않거나 삭제 실패!")

print("✅ 전체 작업 완료: 이미지 이동 + 분할 + 정리 끝!")