import shutil
from pathlib import Path

import kagglehub
from colorama import Fore, Style

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


def main():
    for idx, dataset in enumerate(DATASET_PATHS):
        download_and_copy_dataset(dataset, idx)


if __name__ == '__main__':
    main()
