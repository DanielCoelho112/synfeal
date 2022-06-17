
from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, models

# based on: https://github.com/hazirbas/poselstm-pytorch
# paper: https://openaccess.thecvf.com/content_ICCV_2017/papers/Walch_Image-Based_Localization_Using_ICCV_2017_paper.pdf
class PoseLSTM(nn.Module): 
        def __init__(self, pretrained = True, aux_logits=True):
            super(PoseLSTM, self).__init__()
            
            self.aux_logits = aux_logits

            base_model = models.inception_v3(pretrained)
            base_model.aux_logits = True
            
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
            
            if aux_logits:
                self.aux1 = InceptionAux(288, stride=7)
                self.aux2 = InceptionAux(768, stride=3)
            
            self.lstm_regression = LstmRegression(dropout_rate=0.5)
            

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
            x = self.Mixed_5b(x) # mixed is the inception module!!
            # 35 x 35 x 256
            x = self.Mixed_5c(x)
            # 35 x 35 x 288
            x = self.Mixed_5d(x)
            # 35 x 35 x 288
      
            if self.aux_logits and self.training:
                pose_aux1 = self.aux1(x)
            
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
            
            
            if self.aux_logits and self.training:
                pose_aux2 = self.aux2(x)
            
            
            x = self.Mixed_7a(x)
            # 8 x 8 x 1280
            x = self.Mixed_7b(x)
            # 8 x 8 x 2048
            x = self.Mixed_7c(x)
            # 8 x 8 x 2048
            x = F.avg_pool2d(x, kernel_size=8)
            # 1 x 1 x 2048
            # 1 x 1 x 2048
            x = x.view(x.size(0), -1)
            # 2048
            
            pose = self.lstm_regression(x)
            

            if self.aux_logits and self.training:
                return pose_aux1, pose_aux2, pose
            else:
                return pose
        

class InceptionAux(nn.Module):
    def __init__(self, in_channels, stride):
        super(InceptionAux, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(1,1))
        self.fc = nn.Linear(3200, 2048)        
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=stride)
        self.lstm_regression = LstmRegression(dropout_rate=0.7)
    
    def forward(self, x):
        
        x = self.pool(x)
        x = self.relu(self.conv(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc(x))
        
        pose = self.lstm_regression(x)
        
        return pose

class LstmRegression(nn.Module):
    def __init__(self, dropout_rate, hidden_size = 128):
        super(LstmRegression, self).__init__()
        
        #TODO: try hidden_size = 32
        
        self.hidden_size = hidden_size
        self.lstm_lr = nn.LSTM(input_size=64, hidden_size = hidden_size, bidirectional = True, batch_first = True)
        self.lstm_ud = nn.LSTM(input_size=32, hidden_size = hidden_size, bidirectional = True, batch_first = True)
        
        self.pos = nn.Linear(hidden_size*4, 3, bias=True)
        self.ori = nn.Linear(hidden_size*4, 4, bias=True)
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,x):
        # x is of shape (N,1,2048)
        x = x.view(x.size(0),32, 64)

        _, (hidden_state_lr, _) = self.lstm_lr(x.permute(0,1,2)) # to run row by row
        _, (hidden_state_ud, _) = self.lstm_ud(x.permute(0,2,1)) # to run col by col
        
        # hidden_state_lr.shape = [2, batch_size, hidden_size]
        
        lstm_vector = torch.cat((hidden_state_lr[0,:,:],
                                 hidden_state_lr[1,:,:],
                                 hidden_state_ud[0,:,:],
                                 hidden_state_ud[1,:,:]), 1)
        
        
        lstm_vector = self.dropout(lstm_vector)
        
        pos = self.pos(lstm_vector)
        ori = self.ori(lstm_vector)
            
        pose = torch.cat((pos, ori), dim=1)
        
        
        return pose
        

        
    
# if __name__ == "__main__":
    
#     model = PoseLSTM()
    
#     print(model(torch.rand(10,3,299,299))[0].shape)