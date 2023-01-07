import numpy as np
import torch
import torch.nn as nn

# 損失の分移点


class Quantile_CrossEntropyLoss(nn.Module):
    def __init__(self, q):
        self.q = q
        super(Quantile_CrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels):
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        sorted_loss, _ = torch.sort(loss)
        return torch.quantile(sorted_loss, self.q)


if __name__ == '__main__':
    inputs = torch.randn(100, 5)
    target = torch.empty(100, dtype=torch.long).random_(5)
    criterion = Quantile_CrossEntropyLoss(q=0.5)
    print(criterion.forward(inputs, target))
