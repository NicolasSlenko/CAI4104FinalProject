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
  The main function that orchestrates the data processing pipeline. It creates the transformations, loads and splits the data using a 70/15/15 train/val/test split with batch size 32, and returns the DataLoader objects and the full dataset.

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

- Five convolutional blocks, each consisting of:
  - 2D Convolution layer (from 32 to 512 filters)
  - Batch Normalization
  - ReLU activation
  - Max Pooling (2x2)
  - Dropout (0.1)
- Feature dimensions grow from 32→64→128→256→512 through the convolutional layers
- Fully connected layers:
  - Flatten layer that reshapes 512×4×4 features into 8192-dimensional vector
  - Hidden layer with 516 units
  - Dropout (0.5)
  - Output layer with 12 units (one for each class)

### Data Augmentation

The model implements on-the-fly data augmentation during training that includes:
- Random horizontal flips
- Random vertical flips 
- Random rotations (up to 15 degrees)

### Training Function

The `train_model()` function handles the training process:

- Uses Adam optimizer with learning rate 0.0005 and weight decay 0.0005
- Uses CrossEntropyLoss with label smoothing 0.1
- Implements early stopping with patience=10
- Uses ReduceLROnPlateau scheduler to reduce learning rate by factor of 0.5 when validation accuracy plateaus for 4 consecutive epochs
- Saves the best model during training as `checkpoints/early_stop_model.pth`
- Saves the final model after training as `checkpoints/final_model.pth`
- Tracks and saves training history (loss and accuracy) for later visualization

## Model Training

The `model_train.py` file contains the main function to run the training pipeline:

1. Load and preprocess the data
2. Extract test data to a separate folder for later evaluation
3. Create and compile the model
4. Train the model for up to 100 epochs with early stopping

## Test Data Counter

The `test_data_counter.py` utility allows you to count the number of images per class in the test dataset:

```bash
python test_data_counter.py
```

This script outputs the number of images for each class in the test dataset, which can be useful for understanding the distribution of your evaluation data.

## Learning Curve Visualization

The `plot_learning_curves.py` script allows you to visualize the learning progress of the trained model:

```bash
python plot_learning_curves.py
```

This script:
- Loads training history from the final model's JSON history file
- Creates two visualizations:
  1. **Accuracy Curves**: Plots training and validation accuracy over epochs
  2. **Loss Curves**: Plots training and validation loss over epochs
- Marks and annotates the best validation accuracy and the lowest validation loss
- Saves the visualizations as 'accuracy_curves.png' and 'loss_curves.png'

These visualizations are useful for:
- Identifying potential overfitting or underfitting
- Understanding convergence behaviors
- Analyzing the best points of model performance
- Documentation and reporting

## Evaluation

The `eval.py` script is used to evaluate the trained model:

### Usage

```bash
python eval.py --model_path checkpoints/final_model.pth --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"
```

This script:
1. Loads a trained model from the checkpoint
2. Evaluates it on the test dataset
3. Reports overall test loss and accuracy, as well as per-class accuracy
4. Compares results against a random guessing baseline

### Available Checkpoint

The final model checkpoint is available in the `checkpoints` directory:
- `final_model.pth`: Complete trained model with best performance

### Results

The final model achieves an overall test accuracy of 78.43% compared to the random guessing baseline of 8.68%, which is an improvement of 69.75 percentage points. Different classes show varying performance:

- **High Performance (>85%)**: 
  - Pen (89.36%)
  - Clock (87.72%)  
  - Paper (87.72%)
  - Calculator (87.69%)
  - Keychain (84.75%)

- **Good Performance (70-85%)**:
  - Laptop (80.00%)
  - Water Bottle (80.00%)
  - Book (77.36%)
  - Backpack (76.47%)

- **Lower Performance (<70%)**:
  - Phone (68.49%)
  - Chair (65.00%)
  - Desk (60.00%)

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
4. Visualize learning curves:
   ```bash
   python plot_learning_curves.py
   ``` 