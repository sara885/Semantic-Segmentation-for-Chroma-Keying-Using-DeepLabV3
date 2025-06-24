import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import random


class AIM500Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform["image"](image)
            mask = self.transform["mask"](mask)
        return image, mask


transform = {
    "image": Compose([
        Resize((256, 256)),
        ToTensor(),
       
    ]),
    "mask": Compose([
        Resize((256, 256)),
        ToTensor(),
    ])
}



image_dir = "/content/drive/MyDrive/AIM-500/original"
mask_dir = "/content/drive/MyDrive/AIM-500/trimap"
dataset = AIM500Dataset(image_dir, mask_dir, transform=transform)
dataset = torch.utils.data.Subset(dataset, range(50))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1))


for param in model.backbone.parameters():
    param.requires_grad = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)


def calculate_iou(predicted, target, num_classes=2):
    ious = []
    predicted = torch.argmax(predicted, dim=1)
    for cls in range(num_classes):
        pred = (predicted == cls)
        true = (target == cls)
        intersection = torch.sum(pred & true)
        union = torch.sum(pred | true)
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection.item() / union.item())
    return np.nanmean(ious)


num_epochs = 15
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_iou = 0.0
    num_batches = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.long().squeeze(1).to(device)
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_iou += calculate_iou(outputs, masks)
        num_batches += 1
    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")

print("Training completed!")


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


test_image_dir = "/content/drive/MyDrive/testimages"
test_dataset = TestDataset(test_image_dir, transform=transform["image"])
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)


model.eval()
with torch.no_grad():
    for images in test_dataloader:
        images = images.to(device)
        outputs = model(images)["out"]
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

       
        for i in range(images.size(0)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(images[i].cpu().permute(1, 2, 0))
            axes[0].set_title("Test Image")
            axes[0].axis("off")
            axes[1].imshow(predictions[i], cmap="gray")
            axes[1].set_title("Predicted Mask")
            axes[1].axis("off")
            plt.show()
