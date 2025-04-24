import ignite.engine
import ignite.engine.events
import ignite.metrics
import torch
import time
import ignite
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ignite.handlers as handlers
import numpy as np
import random

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import (
    Flatten,
    Linear,
    Conv2d,
    MaxPool2d,
    Dropout,
    Sequential,
    ReLU,
    BatchNorm2d,
    Dropout2d,
)


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and random.random() < self.p:
            return x.flip(3)  # Flip along width dimension
        return x


class RandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and random.random() < self.p:
            return x.flip(2)  # Flip along height dimension
        return x


class RandomRotation(nn.Module):
    def __init__(self, degrees=10, p=0.5):
        super().__init__()
        self.degrees = degrees
        self.p = p

    def forward(self, x):
        if not self.training or random.random() >= self.p:
            return x

        # Get device of input tensor
        device = x.device

        # Convert angle to tensor and move to correct device
        angle = torch.tensor(
            random.uniform(-self.degrees, self.degrees) * 3.14159 / 180,
            device=device,  # Ensure it's on the same device as x
        )

        # Use tensors all the way
        cos_val = torch.cos(angle)
        sin_val = torch.sin(angle)

        theta = torch.tensor(
            [[cos_val, -sin_val, 0], [sin_val, cos_val, 0]],
            dtype=torch.float,
            device=device,  # Ensure it's on the same device as x
        )

        grid = F.affine_grid(
            theta.unsqueeze(0).repeat(x.size(0), 1, 1), x.size(), align_corners=False
        )

        return F.grid_sample(x, grid, align_corners=False)


def create_compile_model(channel=3, info=False):
    class FPCNN(nn.Module):
        def __init__(self, channel):
            super(FPCNN, self).__init__()
            self.augmentation = nn.Sequential(
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.3),
                RandomRotation(degrees=15, p=0.3),
            )
            self.conv1 = Conv2d(channel, 16, 3, padding="same")
            self.batchnorm1 = BatchNorm2d(16)
            self.pool = MaxPool2d(2)
            self.conv_dropout = Dropout2d(p=0.1)

            self.conv2 = Conv2d(16, 32, 3, padding="same")
            self.batchnorm2 = BatchNorm2d(32)

            self.conv3 = Conv2d(32, 64, 3, padding="same")
            self.batchnorm3 = BatchNorm2d(64)

            self.conv4 = Conv2d(64, 128, 3, padding="same")
            self.batchnorm4 = BatchNorm2d(128)

            self.conv5 = Conv2d(128, 256, 3, padding="same")
            self.batchnorm5 = BatchNorm2d(256)

            self.fc1 = Linear(256 * 4 * 4, 516)
            self.dropout = Dropout(0.5)
            self.fc2 = Linear(516, 12)

        def forward(self, x):
            if self.training:
                x = self.augmentation(x)
            x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
            x = self.conv_dropout(x)
            x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
            x = self.conv_dropout(x)
            x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
            x = self.conv_dropout(x)
            x = self.pool(F.relu(self.batchnorm4(self.conv4(x))))
            x = self.conv_dropout(x)
            x = self.pool(F.relu(self.batchnorm5(self.conv5(x))))
            # print(x.shape)

            x = x.view(-1, 256 * 4 * 4)

            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            # print(x.shape)

            return x
            # print(x.shape)

    model = FPCNN(channel)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
    model.optimizer = optimizer
    model.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)

    return model


def train_model(device, train_loader, val_loader, dataset, max_epochs, patience=10):
    # create and compile model and then move to GPU
    model = create_compile_model()
    model.to(device)
    print(device)

    loss_func = model.loss_func
    optimizer = model.optimizer

    best_val_loss = float("inf")

    scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.5, patience=4)

    os.makedirs("checkpoints", exist_ok=True)
    early_stop_path = "checkpoints/early_stop_model.pth"
    best_model_state = None

    start = time.time()
    for epoch in range(max_epochs):
        model.train()

        train_loss = 0.0
        correct_train_num = 0
        total_train_size = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels[:10])
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, 1)
            correct_train_num += (predictions == labels).sum().item()
            total_train_size += labels.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc = correct_train_num / total_train_size

        model.eval()
        val_loss = 0
        correct_val_num = 0
        total_val_size = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                predictions = torch.argmax(outputs, 1)
                correct_val_num += (predictions == labels).sum().item()
                total_val_size += labels.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct_val_num / total_val_size

        scheduler.step(val_acc)
        print(scheduler.get_last_lr())

        print(
            f"Epoch: {epoch+1} | Training Loss: {train_loss:.4f}, Training Accuracy: {100*train_acc:.2f}% | Validation Loss: {val_loss:.4f} Validation Accuracy: {100*val_acc:.2f}%"
        )
        if val_loss < best_val_loss:
            patience_count = 0
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        else:
            patience_count += 1
            print("worse")
            if patience <= patience_count:
                print(f"Early stopping on epoch {epoch+1}")
                torch.save(
                    model.state_dict(), early_stop_path
                )  # save early stop checkpoint
                break
    end = time.time()

    elapsed_training_time = end - start
    print(f"Training and Validation time: {elapsed_training_time}")
    # Save the model after training is complete
    model_path = "checkpoints/final_model.pth"
    torch.save(model.state_dict(), model_path)
