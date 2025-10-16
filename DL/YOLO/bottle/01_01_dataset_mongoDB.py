from pathlib import Path
import shutil
import os

root = Path(__file__).parent.resolve()

# 원본 디렉터리들
image_dirs = [
    root / 'dataset' / 'origin' / 'train' / 'images',
    root / 'dataset' / 'origin' / 'valid' / 'images'
]
label_dirs = [
    root / 'dataset' / 'origin' / 'train' / 'labels',
    root / 'dataset' / 'origin' / 'valid' / 'labels'
]

# 통합 대상 디렉터리
target_image_dir = root / 'dataset' / 'images'
target_label_dir = root / 'dataset' / 'labels'

# 디렉터리 생성 (이미 있으면 무시)
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

# 이미지 파일 복사
for src_dir in image_dirs:
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(target_image_dir, filename)
        shutil.copy2(src_path, dst_path)

# 라벨 파일 복사
for src_dir in label_dirs:
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(target_label_dir, filename)
        shutil.copy2(src_path, dst_path)

print("📁 이미지와 라벨 파일이 통합 디렉터리로 복사되었습니다!")
