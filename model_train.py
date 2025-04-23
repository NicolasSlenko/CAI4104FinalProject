import torch
import numpy as np
from model_setup import train_model

from data_processing import preprocessing, visualize_class_distribution
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 14})


def main():
    train_loader, val_loader, test_loader, full_dataset = preprocessing()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # visualize_class_distribution(full_dataset)
    train_model(device, train_loader, val_loader, full_dataset, max_epochs=100)


if __name__ == "__main__":
    main()
