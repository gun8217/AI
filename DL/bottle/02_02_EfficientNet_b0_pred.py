import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 📌 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 📁 디렉토리 경로
root = Path(__file__).parent.resolve()
test_dir = root / "dataset/fixed_data_split/test"
model_path = root / "model/EfficientNet_b0/best_all.pt"
save_dir = root / "save"
save_dir.mkdir(parents=True, exist_ok=True)

# 🧱 Letterbox 이미지 리사이즈
def letterbox_image(image, target_size=(256, 256)):
    iw, ih = image.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))
    new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
    return new_image

# 🔍 이미지 전처리
transform = transforms.Compose([
    transforms.Lambda(lambda img: letterbox_image(img, (256, 256))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📦 데이터셋 및 데이터로더
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 📡 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📌 Using device:", device)

# 🧠 EfficientNet 모델 정의
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model = model.to(device)

# 📦 모델 로드
model.load_state_dict(torch.load(model_path))
model.eval()

# 📝 결과 저장 리스트
results = []

# 🧪 예측 및 정확도 평가
correct = 0
total_samples = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total_samples += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 🔍 배치 이미지 파일 경로 수집
        start_idx = len(results)
        batch_paths = [test_dataset.samples[i][0] for i in range(start_idx, start_idx + inputs.size(0))]

        for path, label_idx, pred_idx in zip(batch_paths, labels.cpu().numpy(), predicted.cpu().numpy()):
            results.append({
                'image_path': path,
                'true_label': CLASS_NAMES[label_idx],
                'predicted_label': CLASS_NAMES[pred_idx],
                'result': 'Correct' if label_idx == pred_idx else 'Wrong'
            })

accuracy = correct / total_samples

# ✅ 결과 출력
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Total Test Samples: {total_samples}")

# 💾 CSV 저장
csv_path = save_dir / "test_results_ef.csv"
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['image_path', 'true_label', 'predicted_label', 'result']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"📝 결과 CSV 파일 저장 완료: {csv_path}")

# 📊 혼동 행렬 및 리포트 계산
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
print("\n📋 Classification Report:\n")
print(report)

# 🎨 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_path = save_dir / "confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
print(f"📉 혼동 행렬 이미지 저장 완료: {cm_path}")

# 📄 Classification Report 저장
report_path = save_dir / "classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"📄 분류 리포트 저장 완료: {report_path}")