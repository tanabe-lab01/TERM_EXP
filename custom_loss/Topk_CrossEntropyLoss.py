import numpy as np
import torch
import torch.nn as nn

# 上位 k ％の損失値の平均


class Topk_CrossEntropyLoss(nn.Module):
    def __init__(self, k):
        self.k = k
        super(Topk_CrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels):
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        sorted_loss, idx = torch.sort(loss)
        sorted_loss_lst = list(sorted_loss)[::-1]
        if self.k == 0:
            return torch.tensor(sorted_loss_lst)[0]
        else:
            assert self.k > 0 and self.k <= 1, 'invalid k'
            sorted_loss_lst = sorted_loss_lst[:int(len(loss)*self.k)]
            return torch.tensor(sorted_loss_lst).mean()


if __name__ == '__main__':
    inputs = torch.randn(100, 5)
    target = torch.empty(100, dtype=torch.long).random_(5)
    criterion = Topk_CrossEntropyLoss(k=0)
    print(criterion.forward(inputs, target))
