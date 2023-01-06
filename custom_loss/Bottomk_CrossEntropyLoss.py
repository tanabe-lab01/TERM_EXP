import numpy as np
import torch
import torch.nn as nn

# 下位 k ％の損失値の平均


class Bottomk_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(Bottomk_CrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels, k=0):
        if k == 0:
            return nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
        else:
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            sorted_loss, idx = torch.sort(loss)
            sorted_loss_lst = list(sorted_loss)
            return torch.tensor(sorted_loss_lst).mean()


if __name__ == '__main__':
    inputs = torch.randn(100, 5)
    target = torch.empty(100, dtype=torch.long).random_(5)
    criterion = Bottomk_CrossEntropyLoss()
    print(criterion.forward(inputs, target, k=0.6))
