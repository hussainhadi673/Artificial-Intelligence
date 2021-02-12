
import torch
import numpy as np
import torch.nn as nn


t1=torch.tensor([[4., 7],[3., 5]])
print(t1.dtype)
print(t1)
print(t1.shape)

#Gradient
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

y = w * x + b
print(y)

y.backward()
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)

import numpy as np

x = np.array([[1, 2], [3, 4.]])
print(x)
yy = torch.from_numpy(x)

z=yy.numpy
