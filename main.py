import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision import transforms

# Load your data
images = np.load("data/train_images.npy").astype(np.float32)
masks = np.load("data/train_masks.npy").astype(np.float32)


# Dataset
class CancerSegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Repeat grayscale to 3 channels
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask


# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CancerSegmentationDataset(images, masks, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)

model.classifier[4] = nn.Conv2d(256, 6, kernel_size=1)  # 6 classes
model = model.to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Metrics
def calculate_pixel_accuracy(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    correct = (preds == targets).float()
    acc = correct.sum() / correct.numel()
    return acc


def calculate_simple_pq(preds, targets, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    iou = (intersection + eps) / (union + eps)

    tp = intersection
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    pq = iou * f1
    return pq.mean()


# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_pq = 0.0

    for images_batch, masks_batch in dataloader:
        images_batch = images_batch.to(device)
        masks_batch = masks_batch.to(device)

        optimizer.zero_grad()

        outputs = model(images_batch)['out']

        loss = criterion(outputs, masks_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += calculate_pixel_accuracy(outputs, masks_batch).item()
        running_pq += calculate_simple_pq(outputs, masks_batch).item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_pq = running_pq / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Pixel Acc: {epoch_acc:.4f} | PQ: {epoch_pq:.4f}")

print("Training completed!! 🚀")