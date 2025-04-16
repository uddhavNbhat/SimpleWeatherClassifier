import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset,DataLoader,random_split
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Subset
from collections import defaultdict
import os
import random
import glob
from custom_test_dataset import CustomTestDataset


from weather_model import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ImageFolder("dataset")
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


image_paths = glob.glob("input_test/test_data/*.jpg")

# Create a custom dataset instance
input_dataset = CustomTestDataset(image_paths=image_paths, transform=test_transform)

# Create a DataLoader for the test dataset
dataloader_test = DataLoader(input_dataset,shuffle=False)


model = Net(num_classes=11)
model.load_state_dict(torch.load("weather_model.pth",weights_only=True))
model.to(device)

image_test, label_test = next(iter(dataloader_test))
print(image_test.shape, label_test.shape)

model.eval()

with torch.no_grad():
    for images,_ in dataloader_test:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        class_name = [idx_to_class[int(idx)] for idx in predicted]
        print("Predicted class names:", class_name)

