import torch
import torch.nn as nn
import torch.nn.functional as F
import banana_config as bc


class BananaNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(BananaNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_features=state_size, out_features=bc.num_nodes)
        self.fc2 = nn.Linear(in_features=bc.num_nodes, out_features=bc.num_nodes)
        self.fc3 = nn.Linear(in_features=bc.num_nodes, out_features=bc.num_nodes)
        self.fc4 = nn.Linear(in_features=bc.num_nodes, out_features=action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.fc3(state)
        state = F.relu(state)
        state = self.fc4(state)

        return state
