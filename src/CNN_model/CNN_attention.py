import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN_model.Channel_attention import eca_layer
# from Channel_attention import eca_layer
from CNN_model.Self_attention import Self_Attn

class CNN_Attn(nn.Module):
    
    def __init__(self):
        super(CNN_Attn, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.self_attn1 = eca_layer(3)

        
        self.attn1 = Self_Attn(20, 'relu')
        self.attn2 = Self_Attn(30, 'relu')
        self.fc1 = nn.Linear(750, 1)
        
    def forward(self, x):
        x, channel_x = self.self_attn1(x)
        x = F.relu(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x, p1 = self.attn1(x)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x, p2 = self.attn2(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        out = self.fc1(x)
        return torch.sigmoid(out), p1, p2, channel_x