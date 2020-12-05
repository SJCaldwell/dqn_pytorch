import torch
import torch.nn as nn 
import torch.nn.functional as F

class QNet(nn.Module):
    """Implementation of policy network, mapping state space
       to actions and their rewards"""

    def __init__(self, x, action_space):
       super(QNet, self).__init__()
       self.conv1 = nn.conv2D(in_channels = 4, out_channels=32, kernel_size=4, stride=4)
       self.conv2 = nn.conv2D(in_channels = 32, out_channels=64, kernel_size=4, stride=2)
       self.fc1 = nn.Linear(1024, 256)
       self.fc2 = nn.Linear(256, action_space)
       


    def forward(self, x):
        """Get input data, return output data"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1) # flatten vector
        x = self.fc1(x)
        return self.fc2(x)



        
