import pandas as pd
from pathlib import Path
import shutil

# 경로 설정
yolo_cut_dir = Path("dataset/crops_yolo")
yolo_fixed_dir = Path("dataset/crops_fixed")
csv_path = Path("save/diff_files.csv")

# 1️⃣ 전체 이미지 복사
for class_dir in yolo_cut_dir.iterdir():
    if class_dir.is_dir():
        for img_file in class_dir.glob("*"):
            dest_dir = yolo_fixed_dir / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, dest_dir / img_file.name)

print("✅ 전체 이미지 복사 완료")

# 2️⃣ CSV 기반 덮어쓰기
df = pd.read_csv(csv_path)
origin_class_map = df[['origin', 'class']].dropna().drop_duplicates()
origin_class_dict = dict(zip(origin_class_map['origin'], origin_class_map['class']))
df['matched_class'] = df['yolo'].map(origin_class_dict)

# yolo_fixed_dir 기준으로 파일 경로 매핑
yolo_files_map = {}
for class_dir in yolo_fixed_dir.iterdir():
    if class_dir.is_dir():
        for img_file in class_dir.glob("*"):
            yolo_files_map[img_file.name] = img_file

# matched_class 기준으로 이동
for _, row in df.iterrows():
    file_name = row['yolo']
    target_class = row['matched_class']

    if pd.notna(file_name) and pd.notna(target_class):
        src_path = yolo_files_map.get(file_name)
        dest_dir = yolo_fixed_dir / target_class
        dest_path = dest_dir / file_name

        if src_path and src_path.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(src_path, dest_path)
            print(f"🚚 이동: {file_name} → {dest_dir}")
        else:
            print(f"❌ 파일 없음: {file_name}")

# 3️⃣ _0 제외하고 _1, _2 등 파일 삭제
print("🧹 _0 제외하고 나머지 변형 파일 삭제 중...")
for class_dir in yolo_fixed_dir.iterdir():
    if class_dir.is_dir():
        for img_file in class_dir.glob("*"):
            stem = img_file.stem
            if "_" in stem:
                base, index = stem.rsplit("_", 1)
                if index.isdigit() and index != "0":
                    print(f"🗑 삭제: {img_file.name}")
                    img_file.unlink()

print("✅ 불필요한 변형 이미지 삭제 완료")