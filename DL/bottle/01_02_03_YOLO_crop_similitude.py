import csv
from pathlib import Path

# 디렉토리 설정
origin_dir = Path("dataset/crops_origin")
yolo_cut_dir = Path("dataset/crops")

# 클래스 폴더 리스트 (origin 기준)
class_folders = [p.name for p in origin_dir.iterdir() if p.is_dir()]

rows = []

for cls in class_folders:
    origin_cls_dir = origin_dir / cls
    yolo_cls_dir = yolo_cut_dir / cls

    if not yolo_cls_dir.exists():
        print(f"[경고] yolo_cut_dir에 클래스 폴더 없음: {yolo_cls_dir}")
        continue

    # 파일명 세트 가져오기 (확장자 포함)
    origin_files = set(f.name for f in origin_cls_dir.glob("*"))
    yolo_files = set(f.name for f in yolo_cls_dir.glob("*"))

    # 한쪽에만 있는 파일
    only_in_origin = origin_files - yolo_files
    only_in_yolo = yolo_files - origin_files

    for fname in sorted(only_in_origin):
        rows.append({"class": cls, "origin": fname, "yolo": ""})
    for fname in sorted(only_in_yolo):
        rows.append({"class": cls, "origin": "", "yolo": fname})

# 저장할 경로
csv_path = Path("save/diff_files.csv")
csv_path.parent.mkdir(exist_ok=True)

# CSV 저장
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["class", "origin", "yolo"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ 파일명 차이 결과를 '{csv_path}' 에 저장했습니다.")