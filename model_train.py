import torch
import numpy as np
import os
import shutil
from pathlib import Path
from model_setup import train_model
from torchvision import datasets
from data_processing import preprocessing, visualize_class_distribution
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 14})


def extract_test_data(test_loader, full_dataset):
    """
    Extracts test data to a separate folder for later evaluation.
    """
    # Create test directory
    test_dir = Path("project_test_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)

    # Get class names and create directories
    classes = full_dataset.classes
    for class_name in classes:
        (test_dir / class_name).mkdir(exist_ok=True)

    # Get test dataset
    test_dataset = test_loader.dataset

    # Create a dataset without transforms to get original image paths
    temp_dataset = datasets.ImageFolder(root="./project_data", transform=None)

    # Loop through test dataset and copy files to test directory
    for idx in range(len(test_dataset)):
        global_idx = test_dataset.indices[idx]
        _, label = test_dataset[idx]

        img_path, _ = temp_dataset.samples[global_idx]
        img_filename = os.path.basename(img_path)

        class_name = classes[label]
        dest_path = test_dir / class_name / img_filename
        shutil.copy2(img_path, dest_path)

    extracted_test_set = datasets.ImageFolder(root=str(test_dir))
    print(
        f"Successfully extracted {len(extracted_test_set)} test images to project_test_data/"
    )


def main():
    train_loader, val_loader, test_loader, full_dataset = preprocessing()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # visualize_class_distribution(full_dataset)

    # Extract test data before training
    extract_test_data(test_loader, full_dataset)

    train_model(device, train_loader, val_loader, full_dataset, max_epochs=100)


if __name__ == "__main__":
    main()
