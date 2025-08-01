import cv2
from pathlib import Path
from ultralytics import YOLO  # ultralytics 패키지 설치 필요 (YOLOv8 이상)

# 중복 방지 함수
def get_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    count = 1
    while True:
        new_stem = f"{stem}_{count}"
        new_path = parent / f"{new_stem}{suffix}"
        if not new_path.exists():
            return new_path
        count += 1

# 설정
root = Path(__file__).parent.resolve()
model_path = "runs/bottle3/weights/best.pt"
image_dir = Path(root / "dataset/images")
save_dir = Path(root / "dataset/crops_yolo")
save_dir.mkdir(exist_ok=True)

# 클래스 이름(모델.names로도 가져올 수 있지만, 명확히 쓰려면 직접 정의)
class_map = {
    0: "bad-broken_large",
    1: "bad-broken_small",
    2: "bad-contamination",
    3: "bottle-good"
}

# 클래스별 폴더 생성
for cls_name in class_map.values():
    (save_dir / cls_name).mkdir(exist_ok=True)

# 모델 로드
model = YOLO(model_path)

# 이미지 처리
for img_path in image_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    results = model(str(img_path))  # 이미지 경로 넣어도 됨

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print(f"[{img_path.name}] 탐지된 객체 없음")
        continue

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_idx = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())

        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        crop_img = img[y1:y2, x1:x2]

        cls_name = class_map[cls_idx]

        save_path = save_dir / cls_name / f"{img_path.stem}_{i}{img_path.suffix}"
        save_path = get_unique_path(save_path)  # 중복 검사 및 이름 변경

        cv2.imwrite(str(save_path), crop_img)

print("YOLO 탐지 기반 crop 이미지 저장 완료!")