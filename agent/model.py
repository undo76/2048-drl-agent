import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """QDN model"""

    def __init__(self, state_size, action_size, seed,
                 fc1_units=512, fc2_units=512,
                 v1_units=256, v2_units=64,
                 a1_units=256, a2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.cc1 = nn.Conv2d(16, 128, 3, padding=1)
        self.cc2 = nn.Conv2d(128, 128, 3, padding=1)

        # Common network
        self.fc1 = nn.Linear(4 * 4 * 128, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Duel network - V stream
        self.v1 = nn.Linear(fc1_units, v1_units)
        self.v2 = nn.Linear(v1_units, v2_units)
        self.v3 = nn.Linear(v2_units, 1)

        # Duel network - A stream
        self.a1 = nn.Linear(fc1_units, a1_units)
        self.a2 = nn.Linear(a1_units, a2_units)
        self.a3 = nn.Linear(a2_units, action_size)

    def forward(self, state):
        """Build a Duel network that maps state -> action values."""
        cc1 = F.relu(self.cc1(state))
        cc2 = F.relu(self.cc2(cc1))
        # cc2 = F.dropout(cc2)

        fc1 = F.relu(self.fc1(cc2.view(cc2.size(0), -1)))
        fc1 = F.dropout(fc1)
        # fc2 = F.relu(self.fc2(fc1))
        # fc2 = F.dropout(fc2)

        v1 = F.relu(self.v1(fc1))
        v2 = F.relu(self.v2(v1))
        v3 = self.v3(v2)

        a1 = F.relu(self.a1(fc1))
        a2 = F.relu(self.a2(a1))
        a3 = self.a3(a2)

        # Use the mean in order to keep the identity of the branches
        # as in formula (9) of https://arxiv.org/pdf/1511.06581.pdf
        q = v3 + (a3 - a3.mean(dim=1, keepdim=True))

        return q
