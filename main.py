import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision import transforms
from sklearn.model_selection import KFold
from tqdm import tqdm  # Import tqdm

# EarlyStopping Class
class EarlyStopping:
    """
    Early stops the training if the validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # If validation loss is not a number, skip
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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


# K-Fold Cross Validation
num_epochs = 100
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_metrics = {'loss': [], 'accuracy': [], 'pq': []}

for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    print(f"\nTraining fold {fold + 1}/{k_folds}...")

    # Split the data for this fold
    train_images, val_images = images[train_idx], images[val_idx]
    train_masks, val_masks = masks[train_idx], masks[val_idx]

    # Dataloaders
    train_dataset = CancerSegmentationDataset(train_images, train_masks, transform=transform)
    val_dataset = CancerSegmentationDataset(val_images, val_masks, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Model for this fold
    model = deeplabv3_resnet101(weights=weights)
    model.classifier[4] = nn.Conv2d(256, 6, kernel_size=1)  # 6 classes
    model = model.to(device)

    # Loss and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Early stopping
    early_stopping = EarlyStopping(patience=3, verbose=True, delta=0.001, path=f'checkpoint_fold{fold+1}.pth')

    # Training loop for this fold
    fold_loss = 0.0
    fold_acc = 0.0
    fold_pq = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_pq = 0.0

        # Wrapping the dataloader with tqdm for progress bar
        for images_batch, masks_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100):
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

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        epoch_pq = running_pq / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Pixel Acc: {epoch_acc:.4f} | PQ: {epoch_pq:.4f}")

        fold_loss += epoch_loss
        fold_acc += epoch_acc
        fold_pq += epoch_pq

        # Validation after each epoch
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_pq = 0.0
        with torch.no_grad():
            # Wrapping validation dataloader with tqdm for progress bar
            for images_batch, masks_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", ncols=100):
                images_batch = images_batch.to(device)
                masks_batch = masks_batch.to(device)

                outputs = model(images_batch)['out']
                loss = criterion(outputs, masks_batch)

                val_loss += loss.item()
                val_acc += calculate_pixel_accuracy(outputs, masks_batch).item()
                val_pq += calculate_simple_pq(outputs, masks_batch).item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_pq /= len(val_loader)

        print(f"Validation | Loss: {val_loss:.4f} | Pixel Acc: {val_acc:.4f} | PQ: {val_pq:.4f}")

        # Call early stopping after validation
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break  # Stop training if early stopping criteria are met

    fold_loss /= num_epochs
    fold_acc /= num_epochs
    fold_pq /= num_epochs

    fold_metrics['loss'].append(fold_loss)
    fold_metrics['accuracy'].append(fold_acc)
    fold_metrics['pq'].append(fold_pq)

    print(f"Fold {fold + 1} | Loss: {fold_loss:.4f} | Pixel Acc: {fold_acc:.4f} | PQ: {fold_pq:.4f}")

# Calculate and print overall metrics
avg_loss = np.mean(fold_metrics['loss'])
avg_acc = np.mean(fold_metrics['accuracy'])
avg_pq = np.mean(fold_metrics['pq'])

print("\nK-Fold Cross-Validation Results:")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Average Pixel Accuracy: {avg_acc:.4f}")
print(f"Average PQ: {avg_pq:.4f}")

print("Training completed with K-Fold cross-validation! 🚀")
