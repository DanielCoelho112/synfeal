import torch
from torch import nn

class BetaLoss(nn.Module):
    def __init__(self, beta= 512):
        super(BetaLoss, self).__init__()
        self.beta = beta
        #self.loss_fn = torch.nn.L1Loss() # PoseNet said that L1 was the best
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, pred, targ):
        """
        :param pred: N x 7
        :param targ: N x 7
        :return: 
        """
        # Translation loss
        loss = self.loss_fn(pred[:, :3], targ[:, :3])
        # Rotation loss
        loss += self.beta * self.loss_fn(pred[:, 3:], targ[:, 3:])  ## see paper: https://arxiv.org/abs/1704.00390
        return loss

class TranslationLoss(nn.Module):
    def __init__(self):
        super(TranslationLoss, self).__init__()
        # self.loss_fn = torch.nn.L1Loss() # PoseNet said that L1 was the best
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, pred, targ):
        """
        :param pred: N x 7
        :param targ: N x 7
        :return:
        """
        # Translation loss
        loss = self.loss_fn(pred[:, :3], targ[:, :3])
        # Rotation loss
        # loss += self.beta * self.loss_fn(pred[:, 3:], targ[:, 3:])  ## see paper: https://arxiv.org/abs/1704.00390
        return loss
    
# class DynamicLoss(nn.Module):
#     def __init__(self, sx=0.0, sq=-3.0):
#         super(DynamicLoss, self).__init__()
#         #self.loss_fn = torch.nn.L1Loss() # PoseNet said that L1 was the best
#         self.loss_fn = torch.nn.MSELoss()
#         self.sx = torch.nn.Parameter(torch.Tensor([sx]), requires_grad=True)   # Parameter: When a Parameter is associated with a module as a model attribute, it gets added to the parameter list automatically and can be accessed using the 'parameters' iterator.
#         self.sq = torch.nn.Parameter(torch.Tensor([sq]), requires_grad=True)

        
        
#     def forward(self, pred, targ):
#         """
#         :param pred: N x 7
#         :param targ: N x 7
#         :return: 
#         """
#         # Translation loss
#         loss = torch.exp(-self.sx) * self.loss_fn(pred[:, :3], targ[:, :3]) + self.sx
#         # Rotation loss
#         loss += torch.exp(-self.sq) * self.loss_fn(pred[:, 3:], targ[:, 3:]) + self.sq ## see paper: https://arxiv.org/abs/1704.00390
#         return loss 