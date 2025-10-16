from pathlib import Path
import shutil
import random

# 경로 설정
root = Path(__file__).parent.resolve()
image_dir = root / 'dataset' / 'images'
label_dir = root / 'dataset' / 'labels'

# 대상 디렉토리 생성
for split in ['train', 'val', 'test']:
    (root / 'dataset' / split / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'dataset' / split / 'labels').mkdir(parents=True, exist_ok=True)

# 클래스별 파일 분류
class_files = {key: [] for key in ["0", "1", "2", "3"]}

for label_file in label_dir.glob("*.txt"):
    with label_file.open('r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        class_id = first_line.split()[0]
        if class_id in class_files:
            class_files[class_id].append(label_file)

# 비율 설정
train_ratio = 0.8
val_ratio = 0.15
test_ratio = 0.05

# 클래스별로 균형 있게 분할
for class_id, files in class_files.items():
    random.shuffle(files)
    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split, split_files in splits.items():
        for label_file in split_files:
            base_name = label_file.stem
            image_file = image_dir / f"{base_name}.jpg"

            dst_label = root / 'dataset' / split / 'labels' / label_file.name
            dst_image = root / 'dataset' / split / 'images' / image_file.name

            # 복사
            shutil.copy(label_file, dst_label)
            if image_file.exists():
                shutil.copy(image_file, dst_image)

print("\n✅ 클래스 균형 분할 완료!")