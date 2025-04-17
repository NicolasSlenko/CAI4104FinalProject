import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random

def get_transforms(use_grayscale=False):
    """
    Create image transformations pipeline
    
    Args:
        use_grayscale (bool): Whether to convert images to grayscale
        
    Returns:
        transforms.Compose: The transformation pipeline
    """
    if use_grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  
            transforms.Resize((128, 128)),  
            transforms.ToTensor(),        
            transforms.Normalize((0.5,), (0.5,))  
        ])
        print("Using GRAYSCALE images")
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  
            transforms.ToTensor(),   
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])
        print("Using COLOR images")
    
    return transform

def load_and_split_data(dataset_path, transform, train_size=0.7, val_size=0.15, test_size=0.15, batch_size=32, random_state=42):
    """
    Load dataset and split into training, validation and test sets
    
    Args:
        dataset_path (str): Path to the dataset
        transform (transforms.Compose): Transformations to apply to images
        train_size (float): Proportion of data for training
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        batch_size (int): Batch size for data loaders
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, full_dataset)
    """
   
    if abs(train_size + val_size + test_size - 1.0) > 1e-10:
        raise ValueError("Train, validation, and test sizes must sum to 1")
        
  
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    indices = list(range(len(full_dataset)))
    temp_size = val_size + test_size

    train_indices, temp_indices = train_test_split(
        indices, test_size=temp_size, train_size=train_size, random_state=random_state
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices, 
        test_size=test_size/temp_size,  
        random_state=random_state
    )
    
 
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'Training set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')
    print(f'Test set size: {len(test_dataset)}')
    
    return train_loader, val_loader, test_loader, full_dataset

def visualize_class_distribution(full_dataset):
    """
    Visualize the class distribution in the dataset
    
    Args:
        full_dataset (Dataset): The complete dataset
        
    Returns:
        dict: Counts of images per class
    """
    class_names = full_dataset.classes
    class_counts = {}
    for class_idx in range(len(class_names)):
        class_counts[class_names[class_idx]] = 0
    
    for _, label in full_dataset:
        class_counts[class_names[label]] += 1
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Classes')
    plt.ylabel('Number of images')
    plt.title('Class Distribution in Dataset')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()
    
    return class_counts

def main():
    """Main function to run the data processing pipeline"""
    DATASET_PATH = 'C:/Users/nicks/Downloads/project_data'
    USE_GRAYSCALE = False
    
    transform = get_transforms(use_grayscale=USE_GRAYSCALE)
    BATCH_SIZE = 32
    train_loader, val_loader, test_loader, full_dataset = load_and_split_data(dataset_path=DATASET_PATH,transform=transform,batch_size=BATCH_SIZE)
    
    visualize_class_distribution(full_dataset)
    
    return train_loader, val_loader, test_loader, full_dataset

if __name__ == "__main__":
    # Execute the data processing pipeline when the script is run
    train_loader, val_loader, test_loader, full_dataset = main()
