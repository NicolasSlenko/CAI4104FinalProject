import os
from pathlib import Path


def count_images_per_class(test_data_dir="project_test_data"):
    """Counts the number of images in each class within the given directory.

    Args:
        test_data_dir (str, optional): Path to the directory containing class subfolders. Defaults to "project_test_data".

    Returns:
        dict: A dictionary where keys are class names and values are the number of images in that class.
    """
    test_data_path = Path(test_data_dir)
    if not test_data_path.exists() or not test_data_path.is_dir():
        return f"Error: Directory '{test_data_dir}' not found."

    class_counts = {}
    for class_name in os.listdir(test_data_path):
        class_path = test_data_path / class_name
        if class_path.is_dir():
            image_count = len(
                list(class_path.glob("*"))
            )  # Count files in the directory
            class_counts[class_name] = image_count

    return class_counts


if __name__ == "__main__":
    image_counts = count_images_per_class()

    if isinstance(image_counts, str):
        print(image_counts)  # Print the error message
    else:
        for class_name, count in image_counts.items():
            print(f"Class: {class_name}, Image Count: {count}")
