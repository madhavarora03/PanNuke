import shutil
from pathlib import Path

import kagglehub
import numpy as np
from colorama import Fore, Style
import cv2

DATASET_PATHS = [
    "andrewmvd/cancer-inst-segmentation-and-classification",
    "andrewmvd/cancer-instance-segmentation-and-classification-2",
    "andrewmvd/cancer-instance-segmentation-and-classification-3"
]


def download_and_copy_dataset(dataset: str, idx: int) -> Path:
    """
    Download and copy the dataset to the destination folder.

    Parameters:
        dataset (str): The dataset name from Kaggle.
        idx (int): The index of the dataset.

    Returns:
        Path: The local path to the downloaded dataset.
    """
    try:
        path = kagglehub.dataset_download(dataset)
        print(f"{Fore.GREEN}Path to dataset files for '{dataset}':{Style.RESET_ALL} {path}")

        dest_dir = f"data/fold{idx}"
        shutil.copytree(f"{path}/{'Part 1' if idx == 0 else ''}/Images", dest_dir, dirs_exist_ok=True)
        shutil.copy2(f"{path}/{'Part 1' if idx == 0 else ''}/Masks/masks.npy", dest_dir)
        print(f"{Fore.CYAN}Dataset '{dataset}' copied to '{dest_dir}' directory.{Style.RESET_ALL}")
        return Path(dest_dir)
    except Exception as e:
        print(f"{Fore.RED}Error processing dataset '{dataset}':{Style.RESET_ALL} {e}")
        raise


def apply_zscore_normalization(images, mean, std):
    """
    Apply Z-score normalization to a set of images.

    Parameters:
        images (np.ndarray): Input images array.
        mean (float): Mean value to normalize with.
        std (float): Standard deviation value to normalize with.

    Returns:
        np.ndarray: Normalized images.
    """
    images = images.astype(np.float32)
    normalized_images = (images - mean) / std
    return normalized_images


def process_images(images):
    """
    Process images by applying CLAHE and converting to grayscale.

    Parameters:
        images (np.ndarray): The images to process.

    Returns:
        np.ndarray: The processed images.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_images = []

    for img in images:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        clahe_img = clahe.apply(gray)
        equalized_images.append(clahe_img)

    return np.array(equalized_images)


def process_masks(masks):
    """
    Convert mask images to binary (0 or 1).

    Parameters:
        masks (list): List of mask images to process.

    Returns:
        np.ndarray: The binary masks.
    """
    return np.array([cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1] for mask in masks])


def process_train_data():
    """
    Process the training dataset (fold 0 and fold 1).
    """
    print(f"{Fore.YELLOW}Processing training dataset...{Style.RESET_ALL}")

    # Load training images and masks (fold 0 and fold 1)
    images0 = np.load("data/fold0/images.npy")
    images1 = np.load("data/fold1/images.npy")
    images = np.concatenate((images0, images1), axis=0)

    masks0 = np.load("data/fold0/masks.npy")
    masks1 = np.load("data/fold1/masks.npy")
    masks = np.concatenate((masks0, masks1), axis=0)

    # Process images with CLAHE and convert to grayscale
    images = process_images(images)

    # Compute mean and std on the first image
    mean, std = cv2.meanStdDev(images[0])
    mean = np.round(mean[0][0], 3)
    std = np.round(std[0][0], 3)

    # Apply Z-score normalization
    images = apply_zscore_normalization(images, mean, std)

    # Process masks to binary (0 and 1)
    masks = process_masks(masks)

    # Save processed training data
    Path("data").mkdir(exist_ok=True)
    np.save("data/train_images.npy", images)
    np.save("data/train_masks.npy", masks)

    print(f"{Fore.GREEN}Training data processed and saved.{Style.RESET_ALL}")


def process_test_data():
    """
    Process the test dataset (fold 2).
    """
    print(f"{Fore.YELLOW}Processing test dataset...{Style.RESET_ALL}")

    # Load test images and masks (fold 2)
    images_test = np.load("data/fold2/images.npy")
    masks_test = np.load("data/fold2/masks.npy")

    # Process images with CLAHE and convert to grayscale
    images_test = process_images(images_test)

    # Compute mean and std on the first image of test data (optional)
    mean, std = cv2.meanStdDev(images_test[0])
    mean = np.round(mean[0][0], 3)
    std = np.round(std[0][0], 3)

    # Apply Z-score normalization (use same mean and std as for training)
    images_test = apply_zscore_normalization(images_test, mean, std)

    # Process masks to binary (0 and 1)
    masks_test = process_masks(masks_test)

    # Save processed test data
    Path("data").mkdir(exist_ok=True)
    np.save("data/test_images.npy", images_test)
    np.save("data/test_masks.npy", masks_test)

    print(f"{Fore.GREEN}Test data processed and saved.{Style.RESET_ALL}")


def main():
    for idx, dataset in enumerate(DATASET_PATHS):
        download_and_copy_dataset(dataset, idx)

    # Process training and testing data separately
    process_train_data()
    process_test_data()


if __name__ == '__main__':
    main()
