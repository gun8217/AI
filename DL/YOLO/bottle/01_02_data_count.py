from pathlib import Path
import json
from collections import defaultdict

# 경로 설정
root = Path(__file__).parent.resolve()
target_label_dir = root / 'dataset' / 'labels'

# 카운트 저장용 딕셔너리
class_counts = defaultdict(int)

# 라벨 파일 분석
for label_file in target_label_dir.glob('*.txt'):
    with label_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                class_id = line.split()[0]
                class_counts[class_id] += 1

# 클래스 이름 매핑
class_map = {
    "0": "bad-broken_large",
    "1": "bad-broken_small",
    "2": "bad-contamination",
    "3": "bottle-good"
}

# 사람이 보기 좋은 형태로 변환
output = {
    class_map[class_id]: count for class_id, count in sorted(class_counts.items())
}

# 콘솔 출력
print("클래스별 객체 수:")
for name, count in output.items():
    print(f"{name}: {count}개")

# JSON 저장
# json_path = root / 'save' / 'data_count_before.json'
json_path = root / 'save' / 'data_count_after.json'
with json_path.open('w', encoding='utf-8') as json_file:
    json.dump(output, json_file, ensure_ascii=False, indent=4)

print(f"\n JSON 파일 저장 완료: {json_path}")
