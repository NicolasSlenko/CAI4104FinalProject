import ignite.engine
import ignite.engine.events
import ignite.metrics
import torch
import time
import ignite

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ignite.handlers as handlers
import numpy as np

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Flatten, Linear, Conv2d, MaxPool2d, Dropout, Sequential, ReLU, BatchNorm2d, Dropout2d

def create_compile_model(channel=3, info=False):
    class FPCNN(nn.Module):
        def __init__(self, channel):
            super(FPCNN, self).__init__()
            self.conv1 = Conv2d(channel, 16, 3, stride=2)
            #self.conv1b = Conv2d(16, 16, 3, padding='same')
            self.batchnorm1 = BatchNorm2d(16)
            self.pool1 = MaxPool2d(2)
            self.dropout2d = Dropout2d(0.4)
            self.conv2 = Conv2d(16, 32, 3)
            #self.adaptpool = nn.AdaptiveAvgPool2d(8)
            self.batchnorm2 = BatchNorm2d(32)
            self.conv3 = Conv2d(32, 64, 3)
            self.batchnorm3 = BatchNorm2d(64)

            self.fc1 = Linear(64 * 6 * 6, 512)
            self.dropout = Dropout(0.4)
            #self.fc2 = Linear(512, 128)
            self.fc2 = Linear(512, 12)
        def forward(self, x):
            x = self.pool1(self.dropout2d(self.batchnorm1(F.relu(self.conv1(x)))))
            x = self.pool1(self.dropout2d(self.batchnorm2(F.relu(self.conv2(x)))))
            x = self.pool1(self.dropout2d(self.batchnorm3(F.relu(self.conv3(x)))))
            #x = F.relu(self.batchnorm1(self.conv1(x)))
            #x = self.pool1(F.relu(self.batchnorm1(self.conv1b(x))))
            #x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
            #print(x.shape)
            
            x = x.view(-1, 64 * 6 * 6)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            #x = F.relu(self.fc2(x))
            #x = self.dropout(x)
            #x = self.dropout(x)
            x = self.fc2(x)
            #print(x.shape)

            return x
    
    model = FPCNN(channel)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.optimizer = optimizer
    model.loss_func = nn.CrossEntropyLoss()

    return model

def train_model(device, train_loader, val_loader, dataset, max_epochs, patience=10):
    # create and compile model and then move to GPU
    model = create_compile_model()
    model.to(device)
    print(device)

    loss_func = model.loss_func
    optimizer = model.optimizer

    best_val_loss = float('inf')

    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=4)

    start = time.time()
    for epoch in range(max_epochs):
        model.train()

        train_loss = 0.0
        correct_train_num = 0
        total_train_size = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #print(labels[:10])
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
    
        print(f'Epoch: {epoch+1} | Training Loss: {train_loss:.4f}, Training Accuracy: {100*train_acc:.2f}% | Validation Loss: {val_loss:.4f} Validation Accuracy: {100*val_acc:.2f}%')
        if val_loss < best_val_loss:
            patience_count = 0
            best_val_loss = val_loss
        else:
            patience_count += 1
            print("worse")
            if patience <= patience_count:
              print(f"Early stopping on epoch {epoch+1}")
              break
    end = time.time()

    elapsed_training_time = end-start
    print(f'Training and Validation time: {elapsed_training_time}')