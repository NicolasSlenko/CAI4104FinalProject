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

- **`preprocessing()`**  
  The main function that orchestrates the data processing pipeline. It creates the transformations, loads and splits the data using a 85/7.5/7.5 train/val/test split with batch size 32, and returns the DataLoader objects and the full dataset.

### Usage

To use this module, you need to:

1. Set the `DATASET_PATH` variable in the `preprocessing()` function to point to your dataset directory
2. Run the script to create train, validation, and test data loaders
3. Use these loaders with your PyTorch model for training and evaluation

Example:
```python
from data_processing import preprocessing

# Get data loaders
train_loader, val_loader, test_loader, full_dataset = preprocessing()

# Use loaders with your model
model = YourModel()
# ... training code ...
```

## Model Architecture

The `model_setup.py` file defines the model architecture and training procedure:

### FPCNN Model

The model is a Fully-Parametrized Convolutional Neural Network (FPCNN) with the following architecture:

- Four convolutional blocks, each consisting of:
  - 2D Convolution layer
  - Batch Normalization
  - ReLU activation
  - Max Pooling (2x2)
- Feature dimensions grow from 16 to 128 through the convolutional layers
- Three fully connected layers (1024, 512, 256 units) with dropout (0.5)
- Output layer with 12 units (one for each class)

### Training Function

The `train_model()` function handles the training process:

- Uses Adam optimizer with learning rate 0.0005 and weight decay 0.0005
- Implements early stopping with patience=10
- Uses ReduceLROnPlateau scheduler to reduce learning rate on plateau
- Saves the best model during training as `checkpoints/early_stop_model.pth`
- Saves the final model after training as `checkpoints/final_model.pth`

## Model Training

The `model_train.py` file contains the main function to run the training pipeline:

1. Load and preprocess the data
2. Create and compile the model
3. Train the model for up to 100 epochs with early stopping

## Evaluation

The `eval.py` script is used to evaluate the trained model:

### Usage

```bash
python eval.py --model_path YOUR_SAVED_MODEL --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"
```

This script:
1. Loads a trained model from the checkpoint
2. Evaluates it on the test dataset
3. Reports overall test loss and accuracy, as well as per-class accuracy

### Available Checkpoints

Two model checkpoints are available in the `checkpoints` directory:
- `early_stop_model.pth`: Model saved at the point of early stopping
- `final_model.pth`: Model saved after training completion

## Running the Project

1. Make sure the dataset is placed in the `project_data` directory
2. Run the training script:
   ```bash
   python model_train.py
   ```
3. Evaluate the model:
   ```bash
   python eval.py --model_path checkpoints/final_model.pth --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"
   ``` 