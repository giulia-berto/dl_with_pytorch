#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pandas as pd
from customDataset import HCPanatDataset, CropBorders3D
from cnn import Conv3DNet, training_loop_conv, validate_conv

torch.set_printoptions(edgeitems=2)
torch.manual_seed(0)

#filenames
#src_dir = '../data/HCP-anat-data'
src_dir = '../data/HCP-anat'
img_dir = src_dir + '/images/'
target_file = src_dir + '/annotations.csv'
dataset = HCPanatDataset(csv_file=target_file, root_dir=img_dir)

#hyperparameters
n_crop = 20
perc_train = 0.85
n_epochs = 5
batch_size = 4
learning_rate = 1e-3

#apply some transformation to the data (crop)
transformed_dataset = HCPanatDataset(
    csv_file=target_file, 
    root_dir=img_dir, 
    transform=transforms.Compose([
        CropBorders3D(n_crop)]))

#check dimensions
t1, _ = transformed_dataset[0]
print("Shape of one image after cropping %i slices at the borders:" %n_crop)
print(t1.shape)

#compute the mean and std of the data
max_dim = len(t1.shape) #concatenating dimension
imgs = np.stack([img for img, _ in transformed_dataset], axis=max_dim)
mean = np.mean(imgs)
std = np.std(imgs)
mean, std

#normalize the data
normalized_dataset = HCPanatDataset(
    csv_file=target_file, 
    root_dir=img_dir, 
    transform=transforms.Compose([
        CropBorders3D(n_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)]))

#split the dataset into training and test sets with torch.utils.data.random_split
N = len(normalized_dataset)
train_set, test_set = random_split(normalized_dataset, [int(perc_train*N), N-int(perc_train*N)]) 
print("Total number of images: %i" %N)
print("Number of training images: %i" %(perc_train*N))
print("Number of test images: %i" %(N-int(perc_train*N)))

#infer number of features
n_in = imgs.shape[0] * imgs.shape[1] * imgs.shape[2] #number of input features
labels = pd.read_csv(target_file)['label']
n_out = len(np.unique(labels)) #number of output features, i.e. number of classes
print("The number of input feature is: %i" %n_in)
print("The number of output feature is: %i" %n_out)

#assuming that we are on a CUDA machine, this should print a CUDA device:
device = (torch.device('cuda') if torch.cuda.is_available() 
          else torch.device('cpu'))
print(f"Training on device {device}.")

#define the model, the optimizer and the loss
in_shape3d = t1.shape
conv_model = Conv3DNet(in_shape3d, n_out=n_out).to(device=device)
optimizer = optim.SGD(conv_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

#split the datasets into batches
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#training and showing also validation loss
t0 = time.time()
loss_vector, loss_val_vector = training_loop_conv(
    model = conv_model,
    train_loader = train_loader,
    test_loader = test_loader,
    criterion = loss_fn,
    optimizer = optimizer,
    n_epochs = n_epochs)
print("Training time = %f seconds" %(time.time()-t0))

#plot training and validation loss
plt.figure()
x_axis = np.arange(n_epochs)
plt.plot(x_axis, loss_vector, 'r--', label='loss train')
plt.plot(x_axis, loss_val_vector, 'g--', label='loss val')
plt.ylim(0, 0.2)
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('training_validation_losses.png')

#compute accuracy in training and validation
validate_conv(conv_model, train_loader, test_loader)

numel_list = [p.numel()
              for p in seq_model_large.parameters()
              if p.requires_grad == True]
sum(numel_list), numel_list