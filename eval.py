#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""CAI4104 -- Project -- eval.py

This file contains the evaluation code to load a trained model from a specified checkpoint, evaluate it on a test dataset, and report performance. We will run this script on the test dataset to evaluate the model's performance.
use the command:

python eval.py --model_path YOUR_SAVED_MODEL --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"

Note that you need to write the model creation function and call it in the load_trained_model function. You may also
need to change the predict function (e.g., if your pipeline is not compatible with the provided implementation).
Please test the model creation function and the model loading function and predict function to make sure they work. If we cannot load or evaluate your model, we will apply a penalty.

"""

import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# Import the model creation function from model_setup.py
from model_setup import create_compile_model

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


######### Functions #########


def load_test_dataset(
    data_dir: str, batch_size: int, num_workers: int, image_size: int
):
    """
    Loads the test dataset from a given directory. The directory must contain subfolders
    for each class (like in training). Applies only evaluation transforms.

    Args:
        data_dir (str): Path to test data folder.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        image_size (int): Desired image size.

    Returns:
        DataLoader: DataLoader object for the test dataset.
    """
    test_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader


def load_trained_model(
    model_path: str, num_classes: int, device: str, image_size: int = 128
):
    """
    Builds your model architecture, adjusts the classification head to
    the given number of classes, and loads the trained model weights from a local file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        num_classes (int): Number of output classes. (Should be 12 but left for consistency.)
        device (str): Device for model loading ('cuda' or 'cpu').
        image_size (int): desired input image size. (Not used here but kept for consistency.)

    Returns:
        model: The model loaded on device. (If you are not using pytorch nn.Module directly, it is fine but make sure what it loads is compatible with the rest of the code.)
    """

    model = create_compile_model(channel=3)

    ## Change/rewrite the rest of the function as needed, but make sure what it outputs works with the other functions (e.g., predict)

    # Load local state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move the model to the specified device and set evaluation mode.
    model = model.to(device)
    model.eval()
    return model


def predict(model, x):
    """
    Computes the predicted labels for a batch of input images.

    Args:
        model: Your trained model
        x (torch.Tensor): Input batch of images.

    Returns:
        torch.Tensor: Predicted labels.
    """
    with torch.no_grad():
        outputs = model(x)
        _, y_pred = torch.max(outputs, 1)
    return y_pred


def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset.

    Args:
        model: Your trained model
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device used for evaluation.

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * len(test_loader.dataset.classes)
    class_total = [0] * len(test_loader.dataset.classes)

    random_correct = 0
    random_class_correct = [0] * len(test_loader.dataset.classes)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            random_preds = torch.randint(
                0, len(test_loader.dataset.classes), (labels.size(0),), device=device
            )
            random_correct += (random_preds == labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                pred = preds[i]
                class_correct[label] += (pred == label).item()
                class_total[label] += 1
                random_class_correct[label] += (random_preds[i] == label).item()
                
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    test_loss = running_loss / total
    test_accuracy = correct / total
    random_accuracy = random_correct / total

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(test_loader.dataset.classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            random_class_accuracy = 100 * random_class_correct[i] / class_total[i]
            print(
                f"    {class_name}: {accuracy:.2f}% (Random baseline: {random_class_accuracy:.2f}%)"
            )
        else:
            print(f"    {class_name}: N/A (no test samples)")

    print(f"Model Accuracy: {test_accuracy * 100:.2f}%")
    print(f"\nRandom Baseline Accuracy: {random_accuracy * 100:.2f}%")
    print(
        f"Improvement over Baseline Accuracy: {(test_accuracy - random_accuracy) * 100:.2f} percentage points"
    )
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    all_preds_np = all_preds.numpy()
    all_labels_np = all_labels.numpy()
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))
    
    cm = confusion_matrix(all_labels_np, all_preds_np)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    class_names = test_loader.dataset.classes

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2.0 else "black")

    return test_loss, test_accuracy


######### Main() #########

if __name__ == "__main__":
    exit_code = 0  # reassign a value for errors

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Eval script for CAI4104 project")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (e.g., models/trained_model.pth)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="project_test_data",
        help="Directory containing the test dataset with class subfolders",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for DataLoader"
    )
    parser.add_argument("--image_size", type=int, default=128, help="Input image size")
    parser.add_argument(
        "--group_id",
        type=int,
        required=True,
        help="Project Group ID (non-negative integer)",
    )
    parser.add_argument(
        "--project_title",
        type=str,
        required=True,
        help="Project Title (at least 4 characters)",
    )

    args = parser.parse_args()

    project_group_id = args.group_id
    project_title = args.project_title

    # Validate required parameters.
    assert project_group_id >= 0, "Group ID must be non-negative"
    assert len(project_title) >= 4, "Project title must be at least 4 characters long"

    # Keep track of time.
    st = time.time()

    # Header.
    print(
        "\n---------- [Eval] (Project: {}, Group: {}) ---------".format(
            project_title, project_group_id
        )
    )

    # Determine the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluation device:", device)

    # Load test data.
    test_loader = load_test_dataset(
        args.test_data, args.batch_size, args.num_workers, args.image_size
    )

    # Grab number of classes from the test dataset. (Should be 12)
    num_classes = len(test_loader.dataset.classes)
    print("Number of classes:", num_classes)

    # Load the trained model from the given checkpoint.
    model = load_trained_model(args.model_path, num_classes, device, args.image_size)
    print("Model loaded successfully from:", args.model_path)

    # Evaluate the model on test data.
    test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    print(
        "Test Loss: {:.4f}, Test Accuracy: {:.2f}%".format(
            test_loss, test_accuracy * 100
        )
    )

    # Elapsed time.
    et = time.time()
    elapsed = et - st
    print(
        "---------- [Eval] Elapsed time -- total: {:.1f} seconds ---------\n".format(
            elapsed
        )
    )

    sys.exit(exit_code)
