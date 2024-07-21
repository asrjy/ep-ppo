import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    This MLP will be used as both the actor and the critic.
    in_dim: shape of the observation
    out_dim: number of actions that can be taken
    """
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, out_dim)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output
