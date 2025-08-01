import torch
from torchvision import transforms
from torchvision.models import resnet18, efficientnet_b0
from torch import nn
from PIL import Image

def build_model(device, num_classes, model_name="resnet18"):
    if model_name == "resnet18":
        model = resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    else:
        raise ValueError(f"❌ Unknown model type: {model_name}")

    model = model.to(device)
    model.eval()
    return model

def get_transform():
    def letterbox_image(image, target_size=(256, 256)):
        iw, ih = image.size
        w, h = target_size
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        image_resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', target_size, (128, 128, 128))
        new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    transform = transforms.Compose([
        transforms.Lambda(lambda img: letterbox_image(img, (256, 256))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform