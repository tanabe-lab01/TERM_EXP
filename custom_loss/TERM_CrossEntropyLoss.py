import torch
import torch.nn as nn


class TERM_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(TERM_CrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels, t=0):
        if t == 0:
            return nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
        else:
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            loss = 1/t * torch.log(torch.exp(t * loss).mean())
            return loss
