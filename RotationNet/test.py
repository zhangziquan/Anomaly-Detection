import numpy as np
import torch
import torch.nn.functional as F

x = np.array([[1, 0,1],
              [0, 1,0],
              [1, 0,0]]).astype(np.float32)
y = np.array([1, 1, 0])
weight = np.array([1, 1,1])
x = torch.from_numpy(x)
y = torch.from_numpy(y).long()
weight = torch.from_numpy(weight).float()
loss = F.cross_entropy(x, y, weight=weight, reduction="none")
print(x)
print(y)
print(loss)