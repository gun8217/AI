import pandas as pd
from pathlib import Path
import shutil
import re

# 경로 설정
csv_path = Path("save/diff_files.csv")
yolo_cut_dir = Path("dataset/crops_yolo")
yolo_fixed_dir = Path("dataset/crops_fixed")

# CSV 파일 불러오기
df = pd.read_csv(csv_path)

# 오답(yolo) 이미지를 정답(origin) 위치로 복사
for _, row in df.iterrows():
    class_name = row['class']
    origin = row['origin']
    yolo = row['yolo']

    if pd.notna(origin) and pd.notna(yolo):
        img_name = Path(origin).name
        yolo_img_name = Path(yolo).name
        yolo_img_path = yolo_cut_dir / class_name / yolo_img_name

        # ➤ _숫자 패턴 확인 (예: somefile_1.jpg)
        match = re.search(r'_(\d+)(?=\.[^.]+$)', yolo_img_name)
        if match and match.group(1) != "0":
            print(f"⏭️  Skipped (not _0): {yolo_img_name}")
            continue  # _0이 아닌 경우는 건너뜀

        # 저장 경로
        target_path = yolo_fixed_dir / class_name / img_name
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 복사
        if yolo_img_path.exists():
            shutil.copy2(yolo_img_path, target_path)
            print(f"✅ Copied: {yolo_img_name} → {target_path}")
        else:
            print(f"⚠️ Not found: {yolo_img_path}")

print("🎉 보정 이미지 복사 완료.")