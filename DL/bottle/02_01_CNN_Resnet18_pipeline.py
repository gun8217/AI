import torch
import random
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 📌 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 📁 디렉토리 경로 설정
root = Path(__file__).parent.resolve()
base_dir = root / "dataset/fixed_data_split"
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

# 🔍 이미지 전처리 파이프라인
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

# 🧠 모델 정의 및 출력층 수정
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
EPOCHS = 100
patience = 15
best_val_acc = 0.0
best_val_loss = float('inf')
best_all_val_acc = 0.0
best_all_val_loss = float('inf')
best_train_acc = 0.0
best_train_loss = float('inf')

# 📁 모델 저장 디렉토리
model_dir = root / "model"
model_dir.mkdir(parents=True, exist_ok=True)

# 🚀 학습 루프
patience_counter = 0  # Early Stopping을 위한 카운터

for epoch in range(EPOCHS):
    # 🔧 훈련
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

    # 🧪 검증
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

    # 📊 성능 출력
    print(f"\n📅 Epoch {epoch+1}")
    print(f"🔧 Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"🧪 Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

    # 💾 best_val_cnn 저장 조건: validation 정확도 & 손실이 개선된 경우
    if (
        val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
    ):
        best_val_acc = val_acc
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_dir / "Resnet18/best_val.pt")
        print(f"✅ Saved best_val.pt (Epoch {epoch+1})")
        patience_counter = 0  # 성능 향상되면 카운터 초기화
    else:
        patience_counter += 1  # 성능 향상 없으면 카운터 증가

    # 💾 best_all_cnn 저장 조건: 4개 지표 모두 개선된 경우만 저장
    if (
        (val_acc > best_all_val_acc or (val_acc == best_all_val_acc and val_loss < best_all_val_loss)) and
        (train_acc > best_train_acc or (train_acc == best_train_acc and train_loss < best_train_loss))
    ):
        best_all_val_acc = val_acc
        best_all_val_loss = val_loss
        best_train_acc = train_acc
        best_train_loss = train_loss
        torch.save(model.state_dict(), model_dir / "Resnet18/best_all.pt")
        print(f"✅ Saved best_all.pt (Epoch {epoch+1}) by relaxed all-metric condition")

    # 📅 Early Stopping 조건: patience를 초과한 경우 학습 중단
    if patience_counter >= patience:
        print(f"🚨 Early stopping at Epoch {epoch+1} due to no improvement")
        break

    scheduler.step()

# 💾 최종 모델 저장
torch.save(model.state_dict(), model_dir / "Resnet18/last_model.pt")
print("📦 Final model saved as last_model.pt")