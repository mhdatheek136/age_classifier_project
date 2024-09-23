import torch
import torch.nn as nn

class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # 1024
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 512
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 512
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 128
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128*128*128, 128), 
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 7)  # 7 output classes, since there 7 age classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
