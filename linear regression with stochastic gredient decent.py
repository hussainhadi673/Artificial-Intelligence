import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import TensorDataset
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58],
                   [102, 43, 37], [69, 96, 70], [73, 67, 43],
                   [91, 88, 64], [87, 134, 58], [102, 43, 37],
                   [69, 96, 70], [73, 67, 43], [91, 88, 64],
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133],
                    [22, 37], [103, 119], [56, 70],
                    [81, 101], [119, 133], [22, 37],
                    [103, 119], [56, 70], [81, 101],
                    [119, 133], [22, 37], [103, 119]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

train_ds = TensorDataset(inputs, targets)
#print(train_ds[0:3])
#print(train_ds[[0,3,4]])

batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

#for xb, yb in train_dl:
 # print(yb)
  #print("aaaa", xb)
  #break

model = nn.Linear(3, 2)
#print(list(model.parameters()))

preds = model(inputs)
print("aaa", preds)
loss_fn = F.mse_loss
loss = loss_fn(preds, targets)
print(loss)

opt = torch.optim.SGD(model.parameters(), lr=1e-5)


def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:
            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


fit(200, model, loss_fn, opt, train_dl)

preds = model(inputs)
print(preds)


