import torch
import torch.nn as nn

class transformationResnet(nn.Module):
    def __init__(self, num_classes):
        super(transformationResnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= num_classes, out_channels= 64, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels= 64, out_channels= num_classes, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(size=(1024,2048))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        print(x.shape)

        x = self.conv2(x)
        print(x.shape)
        x = self.up(x)
        print(x.shape)

        return x 
        

