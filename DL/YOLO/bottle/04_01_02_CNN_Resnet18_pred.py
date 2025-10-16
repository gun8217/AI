import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

# 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 디렉토리 경로 설정
root = Path(__file__).parent.resolve()
base_dir = root / "dataset/fixed_data_split"
test_dir = base_dir / "test"

# Letterbox 이미지 리사이즈 함수
def letterbox_image(image, target_size=(256, 256)):
    iw, ih = image.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))
    new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
    return new_image

# 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Lambda(lambda img: letterbox_image(img, (256, 256))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 테스트 데이터셋 & 로더
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 모델 정의 및 출력층 수정
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
)
model = model.to(device)

# 모델 불러오기
model.load_state_dict(torch.load("model/Resnet18/best_all.pt"))
model.eval()

# 정확도 계산 함수
def calculate_accuracy(loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 예측
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # 정확도 계산
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy, total

# 테스트 데이터셋에 대한 정확도 계산
accuracy, total_samples = calculate_accuracy(test_loader)

# 결과 출력
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Total Test Samples: {total_samples}")