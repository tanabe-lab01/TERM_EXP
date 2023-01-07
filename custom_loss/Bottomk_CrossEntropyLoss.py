import numpy as np
import torch
import torch.nn as nn

# 下位 k ％の損失値の平均


class Bottomk_CrossEntropyLoss(nn.Module):
    def __init__(self, k):
        self.k = k
        super(Bottomk_CrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels):
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        loss, _ = torch.sort(loss, descending=False)
        if self.k == 0:
            return loss[0]
        else:
            assert self.k > 0 and self.k <= 1, 'invalid k'
            return loss[:int(len(loss)*self.k)].mean()


if __name__ == '__main__':
    inputs = torch.randn(100, 5)
    target = torch.empty(100, dtype=torch.long).random_(5)
    criterion = Bottomk_CrossEntropyLoss(k=1.0)
    print(criterion.forward(inputs, target))
