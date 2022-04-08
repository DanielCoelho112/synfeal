import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



class CNNDepth(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepth, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
            
            self.fc1 = nn.Linear(25088, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = F.relu(self.conv1(x))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = F.relu(self.conv2(x))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = F.relu(self.conv3(x))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = F.relu(self.conv4(x))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = F.relu(self.conv5(x))
            if verbose: print('layer5 shape ' + str(x.shape))

            x = x.view(x.size(0), -1)
            if verbose: print('x shape ' + str(x.shape))
            # x = F.dropout(x, p=0.5)
            # x = F.relu(self.fc1(x))
            # if verbose: print('fc1 shape ' + str(x.shape))
            #
            # x = F.relu(self.fc2(x))
            # if verbose: print('fc2 shape ' + str(x.shape))
            #
            # x = F.relu(self.fc3(x))
            # if verbose: print('fc3 shape ' + str(x.shape))

            x = F.relu(self.fc1(x))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = F.relu(self.fc2(x))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = F.relu(self.fc3(x))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose
