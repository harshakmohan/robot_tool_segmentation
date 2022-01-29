# Build out the U-Net model here

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and
        assign them as member variables.
        """
        super(UNet, self).__init__()
        self.layer1 = nn.Linear(784,hidden_dim) #784 input features, hidden_dim output features
        self.layer2 = nn.Linear(hidden_dim, 10) #hidden_dim input features, 10 output features
        self.model = nn.Sequential(self.layer1, nn.ReLU(), self.layer2)#, nn.Softmax(dim=1))

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        return self.model(x)