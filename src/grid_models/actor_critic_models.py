import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Utils.actor_critic_utils import fanin_init
import torch

EPS = 0.003

class NormalizedAction():

    def _action(self, action, action_dim):
        act_k = (action_dim[1] - action_dim[0])
        # act_b = (action_dim[1] + action_dim[0])/ 2.
        return np.asarray(act_k * action + action_dim[0])

    def _reverse_action(self, action, action_dim):
        act_k_inv = 2./(action_dim[1] - action_dim[0])
        act_b = (action_dim[1] + action_dim[0])/ 2.
        return np.asarray(act_k_inv * (action - act_b))
    
class ImageActor(nn.Module):

    def __init__(self, action_dim):
        super(ImageActor, self).__init__()

        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(4500, 64)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())


        self.lblfc1 = nn.Linear(1, 64)

        self.fc4 = nn.Linear(64, action_dim)
        self.fc4.weight.data.uniform_(-EPS, EPS)
        self.layer_norm = nn.LayerNorm(256)

        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, state, label):
        B, C, H, W = state.size()
        state = state.view(B, C, H, W)
        s1 = F.relu(F.max_pool2d(self.conv1(state), 2))
        s2 = F.relu(F.max_pool2d(self.conv2(s1), 2))
        B, C, W, H = s2.size()
        s2 = s2.view(B, C*W*H)
        s2 = F.relu(self.fc1(s2))

        lbl_fc1 = F.relu(self.lblfc1(label))
        s2 = s2 + lbl_fc1

        action = self.tanh(self.fc4(s2))
        # action = self.sigmoid(self.fc4(s2))

        return action
    

class ImageCritic(nn.Module):

    def __init__(self, action_dim):
        super(ImageCritic, self).__init__()

        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(4500, 32)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.lblfc1 = nn.Linear(1, 64)
        # self.fcs2 = nn.Linear(256, 128)
        # self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(action_dim, 32)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(64, 1)
        # self.sigmoid = nn.Sigmoid()
        self.fc3.weight.data.uniform_(-EPS, EPS)
        # self.layer_norm = nn.LayerNorm(256)

    
    def forward(self, state, label, action):
        s1 = F.relu(F.max_pool2d(self.conv1(state), 2))
        s2 = F.relu(F.max_pool2d(self.conv2(s1), 2))
        B, C, W, H = s2.size()
        s2 = s2.view(B, C*W*H)
        s2 = F.relu(self.fc1(s2))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim = 1)

        lbl_fc1 = F.relu(self.lblfc1(label))
        x = lbl_fc1 + x
        # x = torch.tanh(self.fc3(x))
        # x = self.sigmoid(self.fc3(x))
        x = self.fc3(x)

        return x
    

