import numpy as np
import glob
import os
import torch
import torchvision.transforms as transforms
from MyVGG import MyVGG


# Load model
loaded_model = torch.load('Arashi_trained_model.h5', map_location=torch.device('cpu'))

# data transform model
transform_data = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((32,32)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ])

def judge(face):
    # transform data
    inp = transform_data(face)
    inp = inp.unsqueeze(0)
    # Predict
    output = loaded_model(inp)
    output = output.to('cpu').detach().numpy()
    output = output[0]
    rank = np.argsort(output)
    output = (np.sort(output))*100
    output = output[::-1]
    output = np.asarray(output, dtype = int)
    output = np.asarray(output, dtype = str)
    rank = rank[::-1]

    return rank, output