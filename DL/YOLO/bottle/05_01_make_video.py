import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
import pandas as pd

# 클래스 정의
names = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 경로 설정
root = Path(__file__).parent.resolve()
image_dir = root / 'dataset/fixed_data_split/test'
csv_path = root / 'save/updated_diff_files.csv'

# 비디오 설정
output_width = 640
output_height = 640
fps = 30
duration_sec = 30
total_frames = fps * duration_sec

# CSV 불러오기
if not csv_path.exists():
    raise FileNotFoundError("matched_result.csv 파일이 없습니다.")
df = pd.read_csv(csv_path)
df['matched_class'] = df['yolo'].map(dict(zip(df['origin'], df['class'])))
matched_dict = dict(zip(df['origin'], df['matched_class']))  # 오답: matched_class 존재

# 오답 이미지 2개 추출
all_wrong = [f for f in matched_dict if matched_dict[f] is not None]
random.shuffle(all_wrong)
selected_wrong = all_wrong[:2]

# 모든 이미지 경로 수집
all_image_paths = []
for class_name in names:
    class_dir = image_dir / class_name
    if class_dir.exists():
        all_image_paths.extend(list(class_dir.glob('*.jpg')))

# 오답 제외하고 랜덤으로 10개 추출 (중복 없음)
remaining_candidates = [p for p in all_image_paths if p.name not in selected_wrong]
random.shuffle(remaining_candidates)
selected_random = remaining_candidates[:10]

# 최종 이미지 목록 (2 오답 + 10 랜덤 = 총 12개)
final_paths = [p for p in all_image_paths if p.name in selected_wrong] + selected_random

# 이미지 처리
resized_imgs = []
for path in final_paths:
    img = cv2.imread(str(path))
    if img is None:
        continue

    img = cv2.resize(img, (output_width, output_height))
    class_name = path.parent.name
    is_wrong = matched_dict.get(path.name, None) is not None

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    pos = (10, output_height - 10)
    color = (0, 0, 255) if is_wrong else (255, 255, 255)  # 빨강 or 흰색

    # 텍스트 삽입
    cv2.putText(img, class_name, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, class_name, pos, font, scale, color, thickness, cv2.LINE_AA)

    resized_imgs.append(img)

# 긴 배너 이미지 생성
if not resized_imgs:
    raise RuntimeError("선택된 이미지가 없습니다.")
long_img = np.hstack(resized_imgs)
long_width = long_img.shape[1]

if long_width <= output_width:
    raise ValueError("이미지 너비가 영상보다 작아 스크롤할 수 없습니다.")

# 비디오 저장
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = root / 'dataset' / 'scroll_label_video.mp4'
video = cv2.VideoWriter(str(video_path), fourcc, fps, (output_width, output_height))

if not video.isOpened():
    raise RuntimeError("VideoWriter 초기화 실패")

# 스크롤 영상 생성
for i in range(total_frames):
    dx = int((long_width - output_width) * (1 - i / total_frames))
    frame = long_img[:, dx:dx + output_width]

    if frame.shape[1] != output_width:
        pad = output_width - frame.shape[1]
        frame = cv2.copyMakeBorder(frame, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=0)

    video.write(frame.astype(np.uint8))

video.release()
print("scroll_label_video.mp4 생성 완료! (2 오답 포함, 총 12장)")