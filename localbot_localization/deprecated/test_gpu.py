#!/usr/bin/env python3

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from colorama import Fore, Style
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime  # to track the time each epoch takes
import argparse
import sys
import os
import yaml
from yaml.loader import SafeLoader
from colorama import Fore
import pickle

from localbot_localization.src.dataset import LocalBotDataset
from localbot_localization.src.loss_functions import BetaLoss, DynamicLoss
from localbot_localization.src.utilities import process_pose
from localbot_localization.src.models.posenet import PoseNetGoogleNet
from localbot_localization.src.torch_utilities import summarizeModel, resumeTraining
from torchvision import transforms



torch.cuda.set_device(0)

a = torch.rand(4)



a.cuda()