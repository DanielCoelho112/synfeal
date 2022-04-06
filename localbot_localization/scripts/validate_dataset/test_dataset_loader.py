#!/usr/bin/env python3

import cv2
import torch.utils.data as data
from localbot_core.src.utilities import read_pcd, matrixToXYZ, matrixToQuaternion
from localbot_localization.src.utilities import normalize_quat
import numpy as np
import torch
import os
import yaml
from yaml.loader import SafeLoader
from localbot_localization.src.dataset import LocalBotDatasetDepth

# pytorch datasets: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

dataset = LocalBotDatasetDepth('seq0_depth_v')
values = dataset[0]
print(len(values))
print(values[0].shape)
print(values[1].shape)
print(values[2].shape)
# print(dataset[0][0].shape)
# print(sum(torch.isnan(dataset[78][0])))

