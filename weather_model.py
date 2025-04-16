import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential( # Feature extractor that consists of convolutional layers and pooling layers.
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), # First convolutional layer that has 32 filters with 3x3 patch size and a window size of 1. Channels = 3 for (R,G,B)
            nn.ReLU(), # Activation function to introduce non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2), # Pooling layer to reduce the spatial dimensions of the feature maps/conv layers while maintaining the most important information.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(), # Flatten the output from the convolutional layers to feed into the fully connected layers.
        )

        self.classifer = nn.Sequential(
            nn.Linear(64*32*32, num_classes), # Fully connected layer that takes the flattened output from the feature extractor and finds probability distribution of the classes.
        )# Classifier that consists of fully connected layers.
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifer(x)
        return x