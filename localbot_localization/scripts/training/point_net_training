#!/usr/bin/env python3

import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from localbot_localization.src.models.pointnet import PointNet, feature_transform_regularizer
from localbot_localization.src.dataset import LocalBotDataset
from localbot_localization.src.loss_functions import BetaLoss, DynamicLoss
from localbot_localization.src.utilities import *
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


## Load the dataset
train_dataset = LocalBotDataset(seq=110, npoints=2000)
test_dataset = LocalBotDataset(seq=111, npoints=2000)
batch_size = 4

## Pytorch data loader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  num_workers=2)
# Num_workers tells the data loader instance how many sub-processes to use for data loading. If the num_worker is zero (default) the GPU has to wait for CPU to load data.

## Build the model
feature_transform = False # lets use feature transform
model = PointNet(feature_transform=feature_transform)

## Optmizer and Loss
#criterion = nn.MSELoss() # TODO: search for a better loss function!! Should we predict the translaction and rotation separately?? RESEARCH!
#criterion = BetaLoss()
criterion = DynamicLoss()

# Add all params for optimization
param_list = [{'params': model.parameters()}]
if isinstance(criterion, DynamicLoss):
    # Add sx and sq from loss function to optimizer params
    param_list.append({'params': criterion.parameters()})

optimizer = optim.Adam(params = param_list, lr=0.001) # the most common optimizer in DL
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # variable learning rate. After 5 epochs, the lr decays 0.5

print(torch.cuda.is_available())
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # if cuda is available use CUDA
#print(device)
#model.to(device)
model.cuda()
#criterion.to(device)
criterion.cuda()

n_epochs = 4
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

for epoch in range(n_epochs):
    t0 = datetime.now()
    scheduler.step() # here we are telling the scheduler that: n_epochs += 1
    train_loss = []
    for i, data in enumerate(train_dataloader):
        points, target = data
        #points.shape --> 2,10000,3
        #target.shape --> 2,6
    
        points = points.transpose(2, 1) # 3xN which is what our network is expecting
        points, target = points.cuda(), target.cuda() # move data into GPU
        
        optimizer.zero_grad() # Clears the gradients of all optimized tensors (always needed in the beginning of the training loop)
        
        model = model.train() # Sets the module in training mode. For example, the dropout module can only be use in training mode.
        
        #print(points.shape)
        pred, trans, trans_feat = model(points) # our model outputs the pose, and the transformations used
        
        pred = process_pose(pred)
            
        loss = criterion(pred, target)
        
        if feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001 ## Regularization! --> Prevent overfitting by adding something to the cost function. The simpler the model the lower the cost function
        
        
        loss.backward() # Computes the gradient of current tensor w.r.t. graph leaves.
        optimizer.step() # Performs a single optimization step (parameter update).
        
        train_loss.append(loss.item())
    train_loss = np.mean(train_loss)
    
    test_loss=[]
    for i, data in enumerate(train_dataloader):
        points, target = data
        points = points.transpose(2, 1) # 3xN which is what our network is expecting
        points, target = points.cuda(), target.cuda() # move data into GPU
        model = model.eval() # Sets the module in evaluation mode.

        pred, _, _ = model(points)
        
        pred = process_pose(pred)
        
        loss = criterion(pred, target)
        
        test_loss.append(loss.item())
    test_loss = np.mean(test_loss)
    

    # save losses
    train_losses[epoch] = train_loss
    test_losses[epoch] = test_loss
    
    dt = datetime.now() - t0
    print(f'epoch {epoch+1}/{n_epochs}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, duration: {dt}')
        
        
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'pointnet.pth')