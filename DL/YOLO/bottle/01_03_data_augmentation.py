from PIL import Image, ImageEnhance
from pathlib import Path
import shutil

# 디렉토리 경로
root = Path(__file__).parent.resolve()
image_dir = root / 'dataset' / 'images'
label_dir = root / 'dataset' / 'labels'

# 증강 함수
def augment_image(img_path):
    img = Image.open(img_path).convert('RGB')
    base_name = img_path.stem
    label_path = label_dir / f"{base_name}.txt"

    augmentations = {
        "vflip": img.transpose(Image.FLIP_TOP_BOTTOM),
        "hflip": img.transpose(Image.FLIP_LEFT_RIGHT),
        "bright": ImageEnhance.Brightness(img).enhance(1.5),
        "color": ImageEnhance.Color(img).enhance(1.8),
    }

    for aug_type, aug_img in augmentations.items():
        aug_name = f"{base_name}_aug_{aug_type}"
        aug_img.save(image_dir / f"{aug_name}.jpg")

        # 라벨 복제
        if label_path.exists():
            shutil.copy(label_path, label_dir / f"{aug_name}.txt")

# 이미지 전체 증강
for img_file in image_dir.glob('*.jpg'):
    augment_image(img_file)

print("✅ 평면 구조 증강 + 라벨 복제 완료!")
