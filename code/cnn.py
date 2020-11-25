from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

device = (torch.device('cuda') if torch.cuda.is_available() 
          else torch.device('cpu'))


class Conv3DNet(nn.Module):
    def __init__(self, in_shape3d, n_chans=16, n_out=2):
        super().__init__()
        self.in_shape3d = in_shape3d
        self.n_chans = n_chans
        self.n_out = n_out
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 8, kernel_size=3, padding=1)
        D = self.in_shape3d[0] // 4
        H = self.in_shape3d[1] // 4
        W = self.in_shape3d[2] // 4
        self.fc1 = nn.Linear(D * H * W * n_chans // 2, 32)
        self.fc2 = nn.Linear(32, self.n_out)
        
    def forward(self, x):
        out = F.max_pool3d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool3d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, self.num_flat_features(out))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features           


def training_loop_conv(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    """Training loop with training and validation loss."""
    loss_vector = np.zeros(n_epochs)
    loss_val_vector = np.zeros(n_epochs)
  
    for epoch in range(n_epochs):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs.unsqueeze(1)) #channels are on dim=1
            loss = criterion(outputs, labels)
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        loss_val = 0.0
        for imgs_test, labels_test in test_loader: 
            imgs_test = imgs_test.to(device=device)
            labels_test = labels_test.to(device=device)
            outputs_test = model(imgs_test.unsqueeze(1)) #channels are on dim=1
            loss_test = criterion(outputs_test, labels_test)
            loss_val += loss_test.item()

        loss_vector[epoch] = float(loss_train/len(train_loader))
        loss_val_vector[epoch] = float(loss_val/len(test_loader))    
        print("Epoch: %d, Training Loss: %f, Validation Loss: %f" 
            %(epoch+1, float(loss_train)/len(train_loader), float(loss_val)/len(test_loader)))    

    return loss_vector, loss_val_vector


def validate_conv(model, train_loader, val_loader):
    """Accuracy in training and in validation."""
    for name, loader in [("train", train_loader), ("validation", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs.unsqueeze(1)) #channels are on dim=1
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name , correct / total))