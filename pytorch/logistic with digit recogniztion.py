import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dataset = MNIST(root='data/', download=True)
test_dataset = MNIST(root='data/', train=False)
print(len(test_dataset))
print(dataset[0])
image, label = dataset[0]
plt.imshow(image, cmap='gray')
#plt.show()
print('Label:', label)

dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())

img_tensor, label = dataset[0]
img_tensor=img_tensor.reshape(-1,784)
#print(img_tensor.shape, label)
#print(img_tensor[:,10:15])
#print(torch.max(img_tensor), torch.min(img_tensor))
#plt.imshow(img_tensor[:,10:200], cmap='gray');
#plt.show()

train_ds, val_ds = random_split(dataset, [50000, 10000])
#print(len(train_ds), len(val_ds))

batch_size=128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = 28*28
num_classes = 10

# Logistic regression model
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = MnistModel()

#print("weight shape", model.linear.weight.shape)
#print("parameters", list(model.parameters()))

for images, labels in train_loader:
    #print(labels)
    #images=images.reshape(-1,784)
    print("dl shape",images.shape)
    outputs = model(images)
    #print("output",outputs)
    #preds = torch.max(outputs, dim=1)
    #print("max",preds)
    break
probs = F.softmax(outputs, dim=1)

# Look at sample probabilities
#print("Sample probabilities:\n", probs[:2].data)

# Add up the probabilities of an output row
#print("Sum: ", torch.sum(probs[0]).item())

max_probs, preds = torch.max(probs, dim=1)
#print(preds)
#print(max_probs)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
#result0 = evaluate(model, val_loader)
#print(result0)

#history1 = fit(75, 0.05, model, train_loader, val_loader)
#torch.save(model, "model.pth")

#print("tuned parameters", list(model.parameters()))
model=torch.load("model.pth")
model.eval()
#print(list(model.parameters()))

test_dataset = MNIST(root='data/',
                train=False,
                transform=transforms.ToTensor())

#for image, label in test_dataset:
   #predi=model(image)
#print(predi)

image, label = test_dataset[45]

#print("label data_test", label)
#print("image", image)
predi=model(image)
probs = F.softmax(predi, dim=1)
max_probs, preds = torch.max(probs, dim=1)
#print("predicted", preds)
#print(max_probs)

import cv2
# Load an color image in grayscale
img = cv2.imread('six1.jpg',0)
img = cv2.resize(img, (28,28))
img = torch.tensor(img).float()
print(img.shape)
for i in range(len(img)):
    img[i] = 255 - img[i]
    img[i] = img[i]/255

print(img)
img= img.reshape(1,28,28)
predi=model(img)
probs = F.softmax(predi, dim=1)
max_probs, preds = torch.max(probs, dim=1)
print("predicted cv2", preds)




