from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

device = (torch.device('cuda') if torch.cuda.is_available() 
          else torch.device('cpu'))
print(f"Training on device {device}.")


class HCPanatDataset(Dataset):
    """HCP anatomical dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = nib.load(img_path).get_fdata().astype(np.float32)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


class CropBorders3D(object):
    """Crop borders of a 3D volume (before converting it to tensor)."""

    def __init__(self, n_crop):
        """
        Args:
            n_crop (int): Number of slices to crop at the top and at
                the bottom of each af the three dimensions.
        """
        assert isinstance(n_crop, int)
        self.n_crop = n_crop

    def __call__(self, image):
        n = self.n_crop
        image = image[n:-n, n:-n, n:-n]

        return image


class SelectAxialSlices3D(object):
    """Select some axial contigous central slices from
       a 3D volume (before converting it to tensor)."""

    def __init__(self, n_axial):
        """
        Args:
            n_axial (int): Number of contigous central slices to keep.
        """
        assert isinstance(n_axial, int)
        self.n_axial = n_axial

    def __call__(self, image):
        z_min = np.int(image.shape[0]/2) - np.int(self.n_axial/2)
        z_max = np.int(image.shape[0]/2) + np.int(self.n_axial/2)
        image = image[:,:,z_min:z_max]

        return image


def training_loop(model, train_loader, criterion, optimizer, n_epochs):
  """Training loop with only training loss."""
  loss_vector = np.zeros(n_epochs)
  
  for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        outputs = model(imgs.view(imgs.shape[0], -1))
        loss = criterion(outputs, labels)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_vector[epoch] = float(loss) 
    print("Epoch: %d, Training Loss: %f" %(epoch, float(loss)))    

  return loss_vector


def training_loop_val(model, train_loader, test_loader, criterion, optimizer, n_epochs):
  """Training loop with training and validation loss."""
  loss_vector = np.zeros(n_epochs)
  loss_val_vector = np.zeros(n_epochs)
  
  for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        outputs = model(imgs.view(imgs.shape[0], -1))
        loss = criterion(outputs, labels)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for imgs_test, labels_test in test_loader: 
        imgs_test = imgs_test.to(device=device)
        labels_test = labels_test.to(device=device)
        outputs_test = model(imgs_test.view(imgs_test.shape[0], -1))
        loss_test = criterion(outputs_test, labels_test)

    loss_vector[epoch] = float(loss)
    loss_val_vector[epoch] = float(loss_test)    
    print("Epoch: %d, Training Loss: %f, Validation Loss: %f" %(epoch, float(loss), float(loss_test)))    

  return loss_vector, loss_val_vector


def check_accuracy(loader, model):
  """Check the accuracy of the model."""
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for imgs, labels in loader:
      imgs = imgs.to(device=device)
      labels = labels.to(device=device)
      outputs = model(imgs.view(imgs.shape[0], -1))
      _, predicted = torch.max(outputs, dim=1)
      total += labels.shape[0]
      correct += int((predicted == labels).sum())
        
  print("Accuracy: %f" % (correct / total))


def validate(model, train_loader, val_loader):
    """Accuracy in training and in validation."""
    for name, loader in [("train", train_loader), ("validation", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs.view(imgs.shape[0], -1))
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name , correct / total))

