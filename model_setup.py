import ignite.engine
import ignite.engine.events
import ignite.metrics
import torch
import time
import ignite
import os
import json

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as transforms
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


def create_compile_model(channel=3, info=False):
    class FPCNN(nn.Module):
        def __init__(self, channel):
            super(FPCNN, self).__init__()
            self.augment = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=15),
                ]
            )
            self.conv1 = Conv2d(channel, 32, 3, padding="same")
            self.batchnorm1 = BatchNorm2d(32)
            self.pool = MaxPool2d(2)
            self.conv_dropout = Dropout2d(p=0.1)

            self.conv2 = Conv2d(32, 64, 3, padding="same")
            self.batchnorm2 = BatchNorm2d(64)

            self.conv3 = Conv2d(64, 128, 3, padding="same")
            self.batchnorm3 = BatchNorm2d(128)

            self.conv4 = Conv2d(128, 256, 3, padding="same")
            self.batchnorm4 = BatchNorm2d(256)

            self.conv5 = Conv2d(256, 512, 3, padding="same")
            self.batchnorm5 = BatchNorm2d(512)

            self.fc1 = Linear(512 * 4 * 4, 516)
            self.dropout = Dropout(0.5)
            self.fc2 = Linear(516, 12)

        def forward(self, x):
            if self.training:
                x = self.augment(x)
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

            x = x.view(-1, 512 * 4 * 4)

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
    final_model_path = "checkpoints/final_model.pth"
    best_model_state = None
    
    # Track training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": []
    }

    start = time.time()
    patience_count = 0
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
        
        #update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epochs"].append(epoch + 1)

        scheduler.step(val_acc)
        print(scheduler.get_last_lr())

        print( 
            f"Epoch: {epoch+1} | Training Loss: {train_loss:.4f}, Training Accuracy: {100*train_acc:.2f}% | Validation Loss: {val_loss:.4f} Validation Accuracy: {100*val_acc:.2f}%"
        )
        if val_loss < best_val_loss:
            patience_count = 0
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            #save the current best model
            torch.save(best_model_state, early_stop_path)
        else:
            patience_count += 1
            print("worse")
            if patience <= patience_count:
                print(f"Early stopping on epoch {epoch+1}")
                break
    end = time.time()

    elapsed_training_time = end - start
    print(f"Training and Validation time: {elapsed_training_time}")
    # Save the model after training is complete
    torch.save(model.state_dict(), final_model_path)
    # Save history for final model
    with open(final_model_path.replace('.pth', '_history.json'), 'w') as f:
        json.dump(history, f)
