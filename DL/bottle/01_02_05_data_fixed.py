import shutil
from pathlib import Path
import random

# 기본 경로
root = Path(__file__).parent.resolve()
src_dir = root / 'dataset' / 'crops_fixed'
dst_base = root / 'dataset' / 'fixed_data_split'

# 분할 비율
split_ratio = {'train': 0.8, 'val': 0.15, 'test': 0.05}

# 클래스별 처리
for class_dir in src_dir.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name
        files = sorted([f for f in class_dir.glob("*") if f.is_file()])
        total = len(files)

        # 셔플
        random.shuffle(files)

        # 개수 계산
        n_train = int(total * split_ratio['train'])
        n_val = int(total * split_ratio['val'])
        n_test = total - n_train - n_val  # 나머지는 test에

        split_files = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:]
        }

        # 파일 이동
        for split, split_list in split_files.items():
            target_dir = dst_base / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in split_list:
                dest_path = target_dir / file_path.name
                shutil.copy2(file_path, dest_path)

        print(f"✅ {class_name}: train={n_train}, val={n_val}, test={n_test}")