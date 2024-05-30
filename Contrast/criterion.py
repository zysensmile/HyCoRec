import torch
from torch import nn


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, gpu):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda(gpu).long()
        loss = self.criterion(x, label)
        return loss
