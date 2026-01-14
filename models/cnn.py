import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, use_batchnorm=True):
        super(CNN, self).__init__()
        
        self.features = nn.ModuleList()
        
        in_channels = 3
        conv_configs = [
            (64, 3, 1, 1),
            (64, 3, 1, 1),
            ('pool', 2, 2),
            (128, 3, 1, 1),
            (128, 3, 1, 1),
            ('pool', 2, 2),
            (256, 3, 1, 1),
            (256, 3, 1, 1),
            (256, 3, 1, 1),
            ('pool', 2, 2),
            (512, 3, 1, 1),
            (512, 3, 1, 1),
            (512, 3, 1, 1),
            ('pool', 2, 2),
        ]
        
        for config in conv_configs:
            if config[0] == 'pool':
                self.features.append(nn.MaxPool2d(kernel_size=config[1], stride=config[2]))
            else:
                out_channels, kernel_size, stride, padding = config
                self.features.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                if use_batchnorm:
                    self.features.append(nn.BatchNorm2d(out_channels))
                self.features.append(nn.ReLU(inplace=True))
                in_channels = out_channels
        
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3, use_batchnorm=True):
        super(SimpleCNN, self).__init__()
        
        layers = []
        
        layers.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Dropout2d(dropout))
        
        layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Dropout2d(dropout))
        
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Dropout2d(dropout))
        
        self.features = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
