import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

# 🧪 시드 고정
SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 📌 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 📁 디렉토리 경로
root = Path(__file__).parent.resolve()
base_dir = root / "dataset/final_split"
train_dir, val_dir, test_dir = base_dir / "train", base_dir / "val", base_dir / "test"

# 🧱 Letterbox 이미지 리사이즈 함수
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

# 📦 데이터셋 & 로더
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 📡 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📌 Using device:", device)

# 🧠 모델 정의
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(CLASS_NAMES))
)
model = model.to(device)

# 🎯 손실함수, 옵티마이저, 스케줄러
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ⏱ 하이퍼파라미터
EPOCHS = 50
patience = 10
patience_counter = 0

best_val_acc = 0.0
best_val_loss = float('inf')
best_all_val_acc = 0.0
best_all_val_loss = float('inf')
best_train_acc = 0.0
best_train_loss = float('inf')

prev_val_acc = None
prev_val_loss = None

# 📁 모델 저장 디렉토리
model_dir = root / "model"
model_dir.mkdir(parents=True, exist_ok=True)

# 📊 로그 저장용 리스트
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# 🚀 학습 루프
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"\n📅 Epoch {epoch+1}")
    print(f"🔧 Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"🧪 Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

    # ⚠️ 이상 에폭 감지 (엄격 기준 적용)
    is_anomaly = False
    if prev_val_acc is not None and prev_val_loss is not None:
        acc_drop = prev_val_acc - val_acc
        loss_spike = val_loss / prev_val_loss if prev_val_loss > 0 else 0
        if acc_drop > 0.02 or loss_spike > 2 or val_loss > 0.2:
            is_anomaly = True
            print(f"⚠️ Skipped saving due to anomaly at Epoch {epoch+1} (Acc drop: {acc_drop:.4f}, Loss spike: {loss_spike:.2f}, Val loss: {val_loss:.4f})")

    # 💾 best_val 저장 조건
    if not is_anomaly and (
        val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
    ):
        best_val_acc = val_acc
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_dir / "Resnet18/best_val.pt")
        print(f"✅ Saved best_val.pt (Epoch {epoch+1})")
        patience_counter = 0
    else:
        patience_counter += 1

    # 💾 best_all 저장 조건
    if not is_anomaly and (
        (val_acc > best_all_val_acc or (val_acc == best_all_val_acc and val_loss < best_all_val_loss)) and
        (train_acc > best_train_acc or (train_acc == best_train_acc and train_loss < best_train_loss))
    ):
        best_all_val_acc = val_acc
        best_all_val_loss = val_loss
        best_train_acc = train_acc
        best_train_loss = train_loss
        torch.save(model.state_dict(), model_dir / "Resnet18/best_all.pt")
        print(f"✅ Saved best_all.pt (Epoch {epoch+1}) by relaxed all-metric condition")

    # 📅 Early Stopping
    if patience_counter >= patience:
        print(f"🚨 Early stopping at Epoch {epoch+1} due to no improvement")
        break

    prev_val_acc = val_acc
    prev_val_loss = val_loss
    scheduler.step()

# 💾 최종 모델 저장
torch.save(model.state_dict(), model_dir / "Resnet18/last_model.pt")
print("📦 Final model saved as last_model.pt")

# 📊 에폭별 성능 시각화
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.plot(train_accuracies, label='Train Accuracy', marker='x')
plt.plot(val_accuracies, label='Validation Accuracy', marker='x')
plt.title('Training & Validation Metrics per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(model_dir / "epoch_metrics.png")
print(f"📈 Saved epoch metrics plot to {model_dir / 'epoch_metrics.png'}")