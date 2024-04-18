# ALL THE PATHS IN THIS FILE SHOULD BE CHANGED TO MATCH YOUR PATH FOR THE TRAINING AND TESTING SETS

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import os
print(os.listdir("D:\\Electron-ETTI\\resurse"))

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

### Train Sets ###

trainset = datasets.ImageFolder("D:\\Electron-ETTI\\resurse\\train-set\\trainss", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, drop_last = True)

# print(len(trainset), len(trainloader))

trainnoisyset = datasets.ImageFolder("D:\\Electron-ETTI\\resurse\\train-set\\trainoisy", transform=transform)
trainnoisyloader = torch.utils.data.DataLoader(trainnoisyset, batch_size=128, shuffle=False, drop_last = True)

#######################################################

### Val sets ###

valset = datasets.ImageFolder("D:\\Electron-ETTI\\resurse\\val-set\\valss", transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size = 128, shuffle = False)

valnoisyset = datasets.ImageFolder("D:\\Electron-ETTI\\resurse\\val-set\\valnoisy", transform=transform)
valloader = torch.utils.data.DataLoader(valnoisyset, batch_size = 128, shuffle = False)

#######################################################





