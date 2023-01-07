import torch
import torch.nn as nn

# TERM版クロスエントロピー


class TERM_CrossEntropyLoss(nn.Module):
    def __init__(self, t):
        self.t = t
        super(TERM_CrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels):
        if self.t == 0:
            return nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
        else:
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            loss = 1/self.t * torch.log(torch.exp(self.t * loss).mean())
            return loss
