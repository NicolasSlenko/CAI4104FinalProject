import os
import shutil
from pathlib import Path
from torchvision import datasets, transforms
from data_processing import preprocessing

test_dir = Path("project_test_data")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(exist_ok=True)

train_loader, val_loader, test_loader, full_dataset = preprocessing()

# Get class names and create directories
classes = full_dataset.classes
for class_name in classes:
    (test_dir / class_name).mkdir(exist_ok=True)

test_dataset = test_loader.dataset

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
print(f"Successfully extracted {len(extracted_test_set)} test images")
    

