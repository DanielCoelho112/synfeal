#!/usr/bin/env python3

import random
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
from localbot_localization.src.models.pointnet import PointNet, feature_transform_regularizer
from localbot_localization.src.dataset import LocalBotDataset
from localbot_localization.src.save_results import SaveResults
from localbot_localization.src.utilities import *
from localbot_core.src.utilities import *
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def main():
    # load the model
    feature_transform = False 
    model = PointNet(feature_transform=feature_transform)
    model.load_state_dict(torch.load('../training/pointnet.pth'))
    model.eval()
    model.cuda()

    # load the testing dataset
    test_dataset = LocalBotDataset(seq=111, npoints=2000)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  num_workers=2)

    # save results class
    results = SaveResults('test1','../training/pointnet.pth', 111 )


    for i, data in enumerate(test_dataloader):
        points, target = data
        points = points.transpose(2, 1) # 3xN which is what our network is expecting
        points= points.cuda() # move data into GPU
        pred, _, _ = model(points)
        pred = process_pose(pred)
        target = target.detach().numpy()
        pred = pred.cpu().detach().numpy()
        
        pos_error = compute_position_error(pred[0], target[0])  # RMSE
        rot_error = compute_rotation_error(pred[0], target[0])
        
        transformation_pred = poseToMatrix(pred[0])
        transformation_target = poseToMatrix(target[0])
        
        results.saveTXT(transformation_target, transformation_pred)
        results.updateCSV(pos_error, rot_error)
        results.step()

    results.saveCSV()
    
    

if __name__ == "__main__":
    main()





    
    
    
 