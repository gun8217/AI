import json
from pathlib import Path
from collections import defaultdict

# 현재 스크립트 기준 경로
root = Path(__file__).parent.resolve()

# 클래스 매핑
class_map = {
    "0": "bad-broken_large",
    "1": "bad-broken_small",
    "2": "bad-contamination",
    "3": "bottle-good"
}

# 최종 결과 저장용
result = {}

for split in ['train', 'val', 'test']:
    label_dir = root / 'dataset' / split / 'labels'
    class_counts = defaultdict(int)
    total_files = 0

    if label_dir.exists():
        for label_file in label_dir.glob('*.txt'):
            total_files += 1
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = line.strip().split()[0]  # 첫 번째 값이 class_id
                        class_counts[class_map.get(class_id, f"unknown({class_id})")] += 1

    result[split] = {
        "total_label_files": total_files,
        "class_counts": dict(class_counts)
    }

# JSON 저장
output_path = root / 'save' / 'model_data_size.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=4)

print(f"model data size saved to {output_path}")