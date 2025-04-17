# Image Classification Project

This project implements an image classification model using PyTorch to classify objects such as backpack, book, calculator, chair, clock, desk, keychain, laptop, paper, pen, phone, and water bottle.

## Data Processing Module

The `data_processing.py` file contains several functions for loading, processing, and visualizing the image dataset:

### Core Functions

- **`get_transforms(use_grayscale=False)`**  
  Creates an image transformation pipeline that resizes images to 128x128 pixels, converts them to PyTorch tensors, and normalizes them. Can optionally convert images to grayscale if the `use_grayscale` parameter is set to True.

- **`load_and_split_data(dataset_path, transform, train_size=0.7, val_size=0.15, test_size=0.15, batch_size=32, random_state=42)`**  
  Loads the dataset from the specified path, applies the given transformations, and splits it into training (70%), validation (15%), and test (15%) sets by default. Returns DataLoader objects for each set, which can be used directly for training and evaluating models.

- **`visualize_class_distribution(full_dataset)`**  
  Creates a bar chart showing the distribution of images across different classes in the dataset, helping to identify any class imbalance issues.

- **`main()`**  
  The main function that orchestrates the data processing pipeline. It creates the transformations, loads and splits the data, and optionally visualizes the class distribution. Returns the DataLoader objects and the full dataset.

### Usage

To use this module, you need to:

1. Set the `DATASET_PATH` variable in the `main()` function to point to your dataset directory
2. Run the script to create train, validation, and test data loaders
3. Use these loaders with your PyTorch model for training and evaluation

Example:
```python
from data_processing import main

# Get data loaders
train_loader, val_loader, test_loader, full_dataset = main()

# Use loaders with your model
model = YourModel()
# ... training code ...
```

### Batch Size

The module uses a batch size of 32 by default, which means 32 images are processed together before updating the model's weights. This provides a good balance between:
- Training speed (larger batches can be processed more efficiently)
- Memory usage (smaller batches require less memory)
- Generalization (smaller batches tend to lead to better generalization)

You can adjust the batch size by changing the `BATCH_SIZE` variable in the `main()` function or by passing a different value to the `batch_size` parameter of `load_and_split_data()`. 