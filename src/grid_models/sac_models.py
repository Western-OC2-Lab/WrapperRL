import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

    
class PatchDropActor(nn.Module):

    def __init__(self, state_dim, action_dim, action_space=None):
        super(PatchDropActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, action_dim)
        

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
            

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        all_output = F.relu(self.output(x))

        return all_output

    
class TwinnedQNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, prev_actions):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNNNetwork(state_dim, action_dim, prev_actions)
        self.Q2 = QNNNetwork(state_dim, action_dim, prev_actions)

    def forward(self, state, label, action, set_indices):

        q1 = self.Q1(state, label, action, set_indices)
        q2 = self.Q2(state, label, action, set_indices)

        return q1, q2
    


class GaussianNNPolicy(nn.Module):

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
        
    def __init__(self, state_dim, action_dim, prev_actions, action_space = None):
        super(GaussianNNPolicy, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.prev_actions = prev_actions

        
        self.fc1 = nn.Linear(self.state_dim + self.prev_actions, 128)
        self.fc2 = nn.Linear(128, 32)

        self.mean = nn.Linear(32, self.action_dim)
        self.log_std = nn.Linear(32, self.action_dim)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
    
    def forward(self, state, label, set_indices):
        set_indices = set_indices.float()
        state = torch.cat((state, set_indices), dim=1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std
    
    def sample(self, state, label, set_indices):
        epsilon = 1e-6
        state = state.float()

        means, log_stds = self.forward(state, label, set_indices)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        x_t = normals.rsample()
        # actions = torch.tanh(xs)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normals.log_prob(x_t)
        # calculate entropies
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(means) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
    

class QNNNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, prev_indices):
        super(QNNNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prev_indices = prev_indices

        self.fc1 = nn.Linear(self.state_dim + self.prev_indices, 128)
        self.fc2 = nn.Linear(128, 32)

        self.act_fc1 = nn.Linear(self.action_dim, 32)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action, label, set_indices):
        set_indices = set_indices.float()
        state = torch.cat((state, set_indices), dim = 1)
        x = F.relu(self.fc1(state))
        state_output = F.relu(self.fc2(x))

        action_output = F.relu(self.act_fc1(action))
        shared_output = torch.cat((state_output, action_output), dim = 1)
        shared_output = self.fc3(shared_output)

        
        return shared_output
    
