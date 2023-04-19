import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# First method, the most general one
class MyVGG(torch.nn.Module):
  # put your models here
  def __init__(self):
    super().__init__()
    self.vgg19_features = models.vgg19(pretrained=False).features
    self.flatten = torch.nn.Flatten()
    self.fc1 = torch.nn.Linear(512, 512)
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=0.2)
    self.fc2 = torch.nn.Linear(512, 256)
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=0.2)
    self.fc3 = torch.nn.Linear(256, 5)
    self.softmax = torch.nn.Softmax(dim=1)
    
  # define inference
  def forward(self,x):
    x = self.vgg19_features(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)
    x = self.softmax(x)

    return x