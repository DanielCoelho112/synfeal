
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms, models

# paper: https://arxiv.org/abs/1703.07971
# github: https://github.com/youngguncho/HourglassPose-Pytorch/blob/master/model.py

class HourglassBatch(nn.Module):
    def __init__(self, pretrained, sum_mode=False, dropout_rate=0.5, aux_logits=False):
        super(HourglassBatch, self).__init__()

        self.sum_mode = sum_mode
        self.dropout_rate = dropout_rate
        self.aux_logits = aux_logits
        if pretrained:
            base_model = models.resnet34('ResNet34_Weights.DEFAULT')
        else:
            base_model = models.resnet34()

        # encoding blocks!
        self.init_block = nn.Sequential(*list(base_model.children())[:4])
        self.res_block1 = base_model.layer1
        self.res_block2 = base_model.layer2
        self.res_block3 = base_model.layer3
        self.res_block4 = base_model.layer4

        # decoding blocks
        if sum_mode:
            self.deconv_block1 = nn.ConvTranspose2d(512, 256, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(256, 128, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(128, 64, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(64, 32, kernel_size=(
                3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        else:
            # concatenation with the encoder feature vectors
            self.deconv_block1 = nn.ConvTranspose2d(512, 256, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(512, 128, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(256, 64, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(128, 32, kernel_size=(
                3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm1d(1024)


        # Regressor
        self.fc_dim_reduce = nn.Linear(56 * 56 * 32, 1024)
        self.fc_trans = nn.Linear(1024, 3)
        self.fc_rot = nn.Linear(1024, 4)

        # Initialize Weights
        init_modules = [self.deconv_block1, self.deconv_block2, self.deconv_block3, self.conv_block,
                        self.fc_dim_reduce, self.fc_trans, self.fc_rot]

        for module in init_modules:
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):

        # conv
        x = self.init_block(x)
        x_res1 = self.res_block1(x)
        x_res2 = self.res_block2(x_res1)
        x_res3 = self.res_block3(x_res2)
        x_res4 = self.res_block4(x_res3)

        # Deconv
        x_deconv1 = self.bn1(F.relu(self.deconv_block1(x_res4)))

        if self.sum_mode:
            x_deconv1 = x_res3 + x_deconv1
        else:
            x_deconv1 = torch.cat((x_res3, x_deconv1), dim=1)

        x_deconv2 = self.bn2(F.relu(self.deconv_block2(x_deconv1)))
        
        if self.sum_mode:
            x_deconv2 = x_res2 + x_deconv2
        else:
            x_deconv2 = torch.cat((x_res2, x_deconv2), dim=1)

        x_deconv3 = self.bn3(F.relu(self.deconv_block3(x_deconv2)))
        
        if self.sum_mode:
            x_deconv3 = x_res1 + x_deconv3
        else:
            x_deconv3 = torch.cat((x_res1, x_deconv3), dim=1)
            
            
        x_conv = self.bn4(F.relu(self.conv_block(x_deconv3)))
        
        x_linear = x_conv.view(x_conv.size(0), -1)
        
        x_linear = self.bn5(F.relu(self.fc_dim_reduce(x_linear)))
        x_linear = F.dropout(x_linear, p=self.dropout_rate,
                             training=self.training)

        position = self.fc_trans(x_linear)
        rotation = self.fc_rot(x_linear)

        x_pose = torch.cat((position, rotation), dim=1)

        return x_pose


