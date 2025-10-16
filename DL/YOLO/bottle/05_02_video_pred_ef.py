import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from utils import build_model, get_transform

CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']
root = Path(__file__).parent.resolve()
cnn_model_path = root / "model/EfficientNet_b0/best_all.pt"
yolo_model_path = root / "runs/bottle3/weights/best.pt"
video_path = root / "dataset/scroll_label_video.mp4"
output_path = root / "dataset/output_video_ef.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

cnn_model = build_model(device=device, num_classes=len(CLASS_NAMES), model_name="efficientnet_b0")
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
cnn_model.eval()
transform = get_transform()

cap = cv2.VideoCapture(str(video_path))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

center_x, center_y = width // 2, height // 2
threshold = 50

yolo = YOLO(str(yolo_model_path))

# 폰트 크기 설정
font_scale_yolo = 1.0
font_scale_gt = 1.0
font_scale_cnn = 1.5

# Ground Truth 라벨 로딩 함수 (YOLO 포맷 → 픽셀 좌표)
def load_yolo_labels(label_path, img_width, img_height):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, cx, cy, w, h = map(float, parts)
            x1 = int((cx - w / 2) * img_width)
            y1 = int((cy - h / 2) * img_height)
            x2 = int((cx + w / 2) * img_width)
            y2 = int((cy + h / 2) * img_height)
            boxes.append((x1, y1, x2, y2, int(class_id)))
    return boxes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = yolo(frame)
    boxes = results[0].boxes
    annotated_frame = frame.copy()

    cnn_label = None
    cnn_shown = False

    # GT 라벨 불러오기
    label_file_name = f"{frame_count:06d}.txt"
    label_path = root / f"data/test/labels/{label_file_name}"
    gt_boxes = load_yolo_labels(label_path, width, height)

    # GT 박스 그리기 (주황색)
    for gx1, gy1, gx2, gy2, gt_class_id in gt_boxes:
        gt_label = CLASS_NAMES[gt_class_id]
        cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 165, 255), 2)
        # GT 텍스트
        cv2.putText(annotated_frame, f"GT: {gt_label}", (gx1, max(gy1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_gt, (0, 165, 255), 2)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        yolo_label = CLASS_NAMES[class_id]

        # YOLO 예측 박스 (파란색)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text_y = max(y1 - 10, 10)

        # YOLO 텍스트
        cv2.putText(annotated_frame, f"YOLO: {yolo_label} ({conf:.2f})",
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_yolo, (255, 0, 0), 2)

        # GT 실제값 찾기
        cx_pred = (x1 + x2) // 2
        cy_pred = (y1 + y2) // 2
        min_dist = float('inf')
        gt_label = None
        for gx1, gy1, gx2, gy2, gt_class_id in gt_boxes:
            cx_gt = (gx1 + gx2) // 2
            cy_gt = (gy1 + gy2) // 2
            dist = (cx_gt - cx_pred) ** 2 + (cy_gt - cy_pred) ** 2
            if dist < min_dist:
                min_dist = dist
                gt_label = CLASS_NAMES[gt_class_id]

        if gt_label:
            cv2.putText(annotated_frame, f"GT: {gt_label}",
                        (x1, text_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_gt, (0, 100, 0), 2)

        # 중심에 위치한 객체 → CNN 판별
        if not cnn_shown and abs(cx_pred - center_x) < threshold and abs(cy_pred - center_y) < threshold:
            cropped = frame[y1:y2, x1:x2]
            img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = cnn_model(input_tensor)
                probs = F.softmax(output, dim=1)
                predicted_idx = torch.argmax(probs, dim=1).item()
                cnn_label = CLASS_NAMES[predicted_idx]

            # CNN 결과 박스 (빨간색)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 중앙 CNN 텍스트
            (text_w, text_h), _ = cv2.getTextSize(cnn_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_cnn, 3)
            text_x = (width - text_w) // 2
            text_y = height // 2
            cv2.putText(annotated_frame, cnn_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_cnn, (0, 0, 255), 3)

            cnn_shown = True

    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"저장 완료: {output_path}")