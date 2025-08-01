import torch
import random
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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
base_dir = root / "dataset/fixed_data_split"
train_dir, val_dir = base_dir / "train", base_dir / "val"

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
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 📡 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📌 Using device:", device)

# 🧠 EfficientNet 모델 정의
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model = model.to(device)

# 🎯 손실함수, 옵티마이저, 스케줄러
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# 🕒 하이퍼파라미터 및 저장 설정
EPOCHS = 100
patience = 15
model_dir = root / "model"
model_dir.mkdir(parents=True, exist_ok=True)

best_val_acc = 0.0
best_all_metric = 0.0
patience_counter = 0

# 🚀 학습 루프
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = train_correct / train_total
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    all_metric = (train_acc + val_acc) / 2

    print(f"\n📅 Epoch {epoch+1}")
    print(f"🔧 Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"🧪 Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

    # ✅ best_val 기준 저장
    improved_val = val_acc > best_val_acc
    if improved_val:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_dir / "EfficientNet_b0/best_val.pt")
        print(f"✅ Saved best_val.pt (Epoch {epoch+1})")

    # ✅ best_all 기준 저장
    improved_all = all_metric > best_all_metric
    if improved_all:  # `best_all_metric` 저장
        best_all_metric = all_metric
        torch.save(model.state_dict(), model_dir / "EfficientNet_b0/best_all.pt")
        print(f"✅ Saved best_all.pt (Epoch {epoch+1}) by relaxed all-metric condition")

    # ⏳ Early stopping 로직
    if improved_val or improved_all:
        patience_counter = 0  # `best_val` 또는 `best_all`이 향상되면 `patience_counter` 리셋
    else:
        patience_counter += 1  # 향상되지 않으면 카운트 증가
        print(f"⏸️ Patience: {patience_counter} / {patience}")

    if patience_counter >= patience:
        print("⛔ Early stopping triggered.")
        break

    scheduler.step()

# 📦 마지막 모델 저장
torch.save(model.state_dict(), model_dir / "EfficientNet_b0/last_model.pt")
print("📦 Final model saved.")