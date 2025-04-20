import torch
from model_setup import train_model
from data_processing import preprocessing, visualize_class_distribution

def main():
    train_loader, val_loader, test_loader, full_dataset = preprocessing()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #visualize_class_distribution(full_dataset)
    train_model(device, train_loader, val_loader, max_epochs=100)

if __name__ == "__main__":
    main()