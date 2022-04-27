import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, models

#https://github.com/youngguncho/PoseNet-Pytorch/blob/6c583a345a20ba17f67b76e54a26cf78e2811604/posenet_simple.py#L119
#https://pytorch.org/hub/pytorch_vision_inception_v3/
class PoseNetGoogleNet(nn.Module): 
        def __init__(self, pretrained,dropout_rate=0.0):
            super(PoseNetGoogleNet, self).__init__()

            base_model = models.inception_v3(pretrained)
            base_model.aux_logits = False
            
            self.Conv2d_1a_3x3 = base_model.Conv2d_1a_3x3
            self.Conv2d_2a_3x3 = base_model.Conv2d_2a_3x3
            self.Conv2d_2b_3x3 = base_model.Conv2d_2b_3x3
            self.Conv2d_3b_1x1 = base_model.Conv2d_3b_1x1
            self.Conv2d_4a_3x3 = base_model.Conv2d_4a_3x3
            self.Mixed_5b = base_model.Mixed_5b
            self.Mixed_5c = base_model.Mixed_5c
            self.Mixed_5d = base_model.Mixed_5d
            self.Mixed_6a = base_model.Mixed_6a
            self.Mixed_6b = base_model.Mixed_6b
            self.Mixed_6c = base_model.Mixed_6c
            self.Mixed_6d = base_model.Mixed_6d
            self.Mixed_6e = base_model.Mixed_6e
            self.Mixed_7a = base_model.Mixed_7a
            self.Mixed_7b = base_model.Mixed_7b
            self.Mixed_7c = base_model.Mixed_7c

            # Out 2
            self.pos = nn.Linear(2048, 3, bias=True)
            self.ori = nn.Linear(2048, 4, bias=True)

        # instead of treating the relu as modules, we can treat them as functions. We can access them via torch funtional
        def forward(self, x, verbose=False):  # this is where we pass the input into the module

            # 299 x 299 x 3
            x = self.Conv2d_1a_3x3(x)
            # 149 x 149 x 32
            x = self.Conv2d_2a_3x3(x)
            # 147 x 147 x 32
            x = self.Conv2d_2b_3x3(x)
            # 147 x 147 x 64
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # 73 x 73 x 64
            x = self.Conv2d_3b_1x1(x)
            # 73 x 73 x 80
            x = self.Conv2d_4a_3x3(x)
            # 71 x 71 x 192
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # 35 x 35 x 192
            x = self.Mixed_5b(x)
            # 35 x 35 x 256
            x = self.Mixed_5c(x)
            # 35 x 35 x 288
            x = self.Mixed_5d(x)
            # 35 x 35 x 288
            x = self.Mixed_6a(x)
            # 17 x 17 x 768
            x = self.Mixed_6b(x)
            # 17 x 17 x 768
            x = self.Mixed_6c(x)
            # 17 x 17 x 768
            x = self.Mixed_6d(x)
            # 17 x 17 x 768
            x = self.Mixed_6e(x)
            # 17 x 17 x 768
            x = self.Mixed_7a(x)
            # 8 x 8 x 1280
            x = self.Mixed_7b(x)
            # 8 x 8 x 2048
            x = self.Mixed_7c(x)
            # 8 x 8 x 2048
            x = F.avg_pool2d(x, kernel_size=8)
            # 1 x 1 x 2048
            x = F.dropout(x, training=self.training)
            # 1 x 1 x 2048
            x = x.view(x.size(0), -1)
            # 2048
            pos = self.pos(x)
            ori = self.ori(x)
            
            x_pose = torch.cat((pos, ori), dim=1)

            return x_pose

