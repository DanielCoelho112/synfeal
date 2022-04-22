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


class CNNDepthLow(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthLow, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
            
            self.fc1 = nn.Linear(18432, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            #self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(1024, 3)
            self.fc_out_rotation = nn.Linear(1024, 4)

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
            
            # x = F.relu(self.fc3(x))
            # if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose




class CNNDepthDropout(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthDropout, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
            
            self.dropout1 = nn.Dropout(p=0.5)
            self.dropout2 = nn.Dropout(p=0.3)
            self.dropout3 = nn.Dropout(p=0.2)
            
            self.fc1 = nn.Linear(25088, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = F.relu(self.droupout3(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = F.relu(self.dropout3(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = F.relu(self.dropout3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = F.relu(self.dropout3(self.conv4(x)))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = F.relu(self.dropout3(self.conv5(x)))
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

            x = F.relu(self.dropout1(self.fc1(x)))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = F.relu(self.dropout2(self.fc2(x)))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = F.relu(self.dropout3(self.fc3(x)))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose



class CNNDepthBatch(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthBatch, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
            
            # Batch norm should be before relu
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(512)
            
            self.bn6 = nn.BatchNorm1d(4096)
            self.bn7 = nn.BatchNorm1d(1024)
            self.bn8 = nn.BatchNorm1d(512)
            
            self.dropout = nn.Dropout(p=0.4)
            
            
            self.fc1 = nn.Linear(25088, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = F.relu(self.bn1(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = F.relu(self.bn2(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = F.relu(self.bn3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = F.relu(self.bn4(self.conv4(x)))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = F.relu(self.bn5(self.conv5(x)))
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

            x = F.relu(self.dropout(self.bn6(self.fc1(x))))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = F.relu(self.bn7(self.fc2(x)))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = F.relu(self.bn8(self.fc3(x)))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose
        
        
class CNNDepthBatchK3(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthBatchK3, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
            
            # Batch norm should be before relu
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(512)
            
            self.bn6 = nn.BatchNorm1d(4096)
            self.bn7 = nn.BatchNorm1d(1024)
            self.bn8 = nn.BatchNorm1d(512)
            
            self.dropout = nn.Dropout(p=0.4)
            
            
            self.fc1 = nn.Linear(25088, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=True):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = F.relu(self.bn1(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = F.relu(self.bn2(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = F.relu(self.bn3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = F.relu(self.bn4(self.conv4(x)))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = F.relu(self.bn5(self.conv5(x)))
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

            x = F.relu(self.dropout(self.bn6(self.fc1(x))))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = F.relu(self.bn7(self.fc2(x)))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = F.relu(self.bn8(self.fc3(x)))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose
        
        
class CNNDepthBatchLeaky(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthBatchLeaky, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
            
            # Batch norm should be before relu
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(512)
            
            self.bn6 = nn.BatchNorm1d(4096)
            self.bn7 = nn.BatchNorm1d(1024)
            self.bn8 = nn.BatchNorm1d(512)
            
            self.lrelu = nn.LeakyReLU(0.1)
            
            
            self.fc1 = nn.Linear(25088, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = self.lrelu(self.bn1(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = self.lrelu(self.bn2(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = self.lrelu(self.bn3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = self.lrelu(self.bn4(self.conv4(x)))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = self.lrelu(self.bn5(self.conv5(x)))
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

            x = self.lrelu(self.dropout(self.bn6(self.fc1(x))))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = self.lrelu(self.bn7(self.fc2(x)))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = self.lrelu(self.bn8(self.fc3(x)))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose
        
        
        
class CNNDepthBatchLow(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthBatchLow, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
            self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
            
            # Batch norm should be before relu
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm2d(512)
            
            self.bn6 = nn.BatchNorm1d(4096)
            self.bn7 = nn.BatchNorm1d(1024)
            #self.bn8 = nn.BatchNorm1d(512)
            
            #self.lrelu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(p=0.5)
            
            
            self.fc1 = nn.Linear(25088, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            #self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(1024, 3)
            self.fc_out_rotation = nn.Linear(1024, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=True):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = F.relu(self.bn1(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = F.relu(self.bn2(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = F.relu(self.bn3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = F.relu(self.bn4(self.conv4(x)))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = F.relu(self.bn5(self.conv5(x)))
            if verbose: print('layer5 shape ' + str(x.shape))

            x = x.view(x.size(0), -1)
            if verbose: print('x shape ' + str(x.shape))
            # x = F.dropout(x, p=0.5)
            # x = self.lrelu(self.fc1(x))
            # if verbose: print('fc1 shape ' + str(x.shape))
            #
            # x = self.lrelu(self.fc2(x))
            # if verbose: print('fc2 shape ' + str(x.shape))
            #
            # x = self.lrelu(self.fc3(x))
            # if verbose: print('fc3 shape ' + str(x.shape))

            x = F.relu(self.dropout(self.bn6(self.fc1(x))))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = F.relu(self.bn7(self.fc2(x)))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            # x = self.lrelu(self.bn8(self.fc3(x)))
            # if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose
        
class CNNDepthBatchLowL2RegLeaky(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthBatchLowL2RegLeaky, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=3, padding=1)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=3, padding=1)
            #self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=3, padding=1)
            #self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=3, padding=1)
            
            # Batch norm should be before relu
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            #self.bn4 = nn.BatchNorm2d(256)
            #self.bn5 = nn.BatchNorm2d(512)
            
            self.bn6 = nn.BatchNorm1d(4096)
            self.bn7 = nn.BatchNorm1d(1024)
            self.bn8 = nn.BatchNorm1d(512)
            
            self.lrelu = nn.LeakyReLU(0.2)
            #self.dropout = nn.Dropout(p=0.5)
            
            
            self.fc1 = nn.Linear(20736, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = self.lrelu(self.bn1(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = self.lrelu(self.bn2(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = self.lrelu(self.bn3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            # x = self.lrelu(self.bn4(self.conv4(x)))
            # if verbose: print('layer4 shape ' + str(x.shape))
            # x = self.lrelu(self.bn5(self.conv5(x)))
            # if verbose: print('layer5 shape ' + str(x.shape))

            x = x.view(x.size(0), -1)
            if verbose: print('x shape ' + str(x.shape))
            # x = F.dropout(x, p=0.5)
            # x = self.lrelu(self.fc1(x))
            # if verbose: print('fc1 shape ' + str(x.shape))
            #
            # x = self.lrelu(self.fc2(x))
            # if verbose: print('fc2 shape ' + str(x.shape))
            #
            # x = self.lrelu(self.fc3(x))
            # if verbose: print('fc3 shape ' + str(x.shape))

            x = self.lrelu(self.bn6(self.fc1(x)))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = self.lrelu(self.bn7(self.fc2(x)))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = self.lrelu(self.bn8(self.fc3(x)))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose
        
        
        

class CNNDepthBatchLowL2Reg2(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthBatchLowL2Reg2, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=1)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=1)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=1)
            #self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=1)
            
            # Batch norm should be before relu
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(512)
            #self.bn6 = nn.BatchNorm2d(1024)
            
            
            self.bn6 = nn.BatchNorm1d(4096)
            self.bn7 = nn.BatchNorm1d(1024)
            self.bn8 = nn.BatchNorm1d(512)
            
            #self.lrelu = nn.LeakyReLU(0.2)
            #self.dropout = nn.Dropout(p=0.5)
            
            
            self.fc1 = nn.Linear(18432, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = F.relu(self.bn1(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = F.relu(self.bn2(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = F.relu(self.bn3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = F.relu(self.bn4(self.conv4(x)))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = F.relu(self.bn5(self.conv5(x)))
            if verbose: print('layer5 shape ' + str(x.shape))
            # x = self.lrelu(self.bn6(self.conv6(x)))
            # if verbose: print('layer6 shape ' + str(x.shape))

            x = x.view(x.size(0), -1)
            if verbose: print('x shape ' + str(x.shape))
            # x = F.dropout(x, p=0.5)
            # x = self.lrelu(self.fc1(x))
            # if verbose: print('fc1 shape ' + str(x.shape))
            #
            # x = self.lrelu(self.fc2(x))
            # if verbose: print('fc2 shape ' + str(x.shape))
            #
            # x = self.lrelu(self.fc3(x))
            # if verbose: print('fc3 shape ' + str(x.shape))

            x = F.relu(self.bn6(self.fc1(x)))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = F.relu(self.bn7(self.fc2(x)))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = F.relu(self.bn8(self.fc3(x)))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose
        
        
        

class CNNDepthBatchDropout8(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthBatchDropout8, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
            
            # Batch norm should be before relu
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(512)
            
            self.bn6 = nn.BatchNorm1d(4096)
            self.bn7 = nn.BatchNorm1d(1024)
            self.bn8 = nn.BatchNorm1d(512)
            
            self.dropout = nn.Dropout(p=0.8)
            
            
            self.fc1 = nn.Linear(25088, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = F.relu(self.bn1(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = F.relu(self.bn2(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = F.relu(self.bn3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = F.relu(self.bn4(self.conv4(x)))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = F.relu(self.bn5(self.conv5(x)))
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

            x = F.relu(self.dropout(self.bn6(self.fc1(x))))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = F.relu(self.bn7(self.fc2(x)))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = F.relu(self.bn8(self.fc3(x)))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose
        
        
        
        
        
class CNNDepthBatchDropoutVar(nn.Module): #https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
        def __init__(self):
            super(CNNDepthBatchDropoutVar, self).__init__()  # call the init constructor of the nn.Module. This way, we are only adding attributes.

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
            
            # Batch norm should be before relu
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(512)
            
            self.bn6 = nn.BatchNorm1d(4096)
            self.bn7 = nn.BatchNorm1d(1024)
            self.bn8 = nn.BatchNorm1d(512)
            
            self.dropout1 = nn.Dropout(p=0.5)
            self.dropout1 = nn.Dropout(p=0.3)
            self.dropout1 = nn.Dropout(p=0.2)
            
            
            self.fc1 = nn.Linear(25088, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc_out_translation = nn.Linear(512, 3)
            self.fc_out_rotation = nn.Linear(512, 4)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            if verbose: print('shape ' + str(x.shape))
            x = F.relu(self.bn1(self.conv1(x)))
            if verbose: print('layer1 shape ' + str(x.shape))
            x = F.relu(self.bn2(self.conv2(x)))
            if verbose: print('layer2 shape ' + str(x.shape))
            x = F.relu(self.bn3(self.conv3(x)))
            if verbose: print('layer3 shape ' + str(x.shape))
            x = F.relu(self.bn4(self.conv4(x)))
            if verbose: print('layer4 shape ' + str(x.shape))
            x = F.relu(self.bn5(self.conv5(x)))
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

            x = F.relu(self.dropout1(self.bn6(self.fc1(x))))
            if verbose: print('fc1 shape ' + str(x.shape))

            x = F.relu(self.dropout2(self.bn7(self.fc2(x))))
            if verbose: print('fc2 shape ' + str(x.shape))
            
            x = F.relu(self.dropout3(self.bn8(self.fc3(x))))
            if verbose: print('fc3 shape ' + str(x.shape))

            x_translation = self.fc_out_translation(x)
            if verbose: print('x_translation shape ' + str(x_translation.shape))

            x_rotation = self.fc_out_rotation(x)
            if verbose: print('x_rotation shape ' + str(x_rotation.shape))

            x_pose = torch.cat((x_translation, x_rotation), dim=1)

            return x_pose