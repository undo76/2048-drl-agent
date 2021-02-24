from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricConv2d(nn.Module):

    def __init__(self, conv2d: nn.Conv2d):
        super(SymmetricConv2d, self).__init__()
        self.conv2d = conv2d
        self.in_channels = self.conv2d.in_channels
        self.out_channels = self.conv2d.out_channels
        self.kernel_size = self.conv2d.kernel_size
        self.stride = self.conv2d.stride
        self.padding = self.conv2d.padding
        self.weight, self.bias = self.conv2d.weight, self.conv2d.bias

    def forward(self, input):
        w = self.weight
        s1 = self.conv2d(input)
        s2 = F.conv2d(input, weight=w.flip(dims=(2,)), bias=self.bias, stride=self.stride, padding=self.padding)
        s3 = F.conv2d(input, weight=w.flip(dims=(3,)), bias=self.bias, stride=self.stride, padding=self.padding)
        s4 = F.conv2d(input, weight=w.flip(dims=(2, 3)), bias=self.bias, stride=self.stride, padding=self.padding)

        wt = w.permute(0, 1, 3, 2)
        s1t = F.conv2d(input, weight=wt, bias=self.bias, stride=tuple(reversed(self.stride)),
                       padding=tuple(reversed(self.padding)))
        s2t = F.conv2d(input, weight=wt.flip(dims=(2,)), bias=self.bias, stride=tuple(reversed(self.stride)),
                       padding=tuple(reversed(self.padding)))
        s3t = F.conv2d(input, weight=wt.flip(dims=(3,)), bias=self.bias, stride=tuple(reversed(self.stride)),
                       padding=tuple(reversed(self.padding)))
        s4t = F.conv2d(input, weight=wt.flip(dims=(2, 3)), bias=self.bias, stride=tuple(reversed(self.stride)),
                       padding=tuple(reversed(self.padding)))

        return torch.cat((s1, s2, s3, s4, s1t, s2t, s3t, s4t), dim=1)


class SymmetricRowConv2d(nn.Module):

    def __init__(self, conv2d: nn.Conv2d):
        super(SymmetricRowConv2d, self).__init__()
        self.conv2d = conv2d
        self.in_channels = self.conv2d.in_channels
        self.out_channels = self.conv2d.out_channels
        self.kernel_size = self.conv2d.kernel_size
        self.stride = self.conv2d.stride
        self.padding = self.conv2d.padding
        self.weight, self.bias = self.conv2d.weight, self.conv2d.bias

    def forward(self, input):
        w = self.weight
        s1 = self.conv2d(input)
        s2 = F.conv2d(input, weight=w.flip(dims=(2,)), bias=self.bias, stride=self.stride, padding=self.padding)

        wt = w.permute(0, 1, 3, 2)
        s1t = F.conv2d(input, weight=wt, bias=self.bias, stride=tuple(reversed(self.stride)),
                       padding=tuple(reversed(self.padding)))
        s2t = F.conv2d(input, weight=wt.flip(dims=(3,)), bias=self.bias, stride=tuple(reversed(self.stride)),
                       padding=tuple(reversed(self.padding)))

        return torch.cat((s1, s2, s1t.permute(0, 1, 3, 2), s2t.permute(0, 1, 3, 2)), dim=1)


class QNetwork(nn.Module):
    """QDN model"""

    def __init__(self, action_size, seed,
                 v1_units=256, v2_units=256,
                 a1_units=256, a2_units=256):
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

        self.emb = nn.Embedding(16, 16, padding_idx=0)

        self.cc1 = nn.Sequential(
            SymmetricConv2d(nn.Conv2d(16, 16, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(16 * 8),
            nn.ReLU(),
            # nn.Dropout2d()
        )
        # self.cc2 = SymmetricConv2d(nn.Conv2d(64, 16, 4))
        self.cc3 = nn.Sequential(
            SymmetricConv2d(nn.Conv2d(16 * 8, 16, 4)),
            # nn.BatchNorm2d(16 * 8),
            nn.ReLU(),
            # nn.Dropout2d()
        )

        self.cc_rows = nn.Sequential(
            SymmetricRowConv2d(nn.Conv2d(16, 16, (1, 4))),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d()
        )
        # self.cc_cols = nn.Sequential(
        #     nn.Conv2d(16, 32, (4, 1)),
        #     # nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     # nn.Dropout2d()
        # )

        # Common network
        self.fc1 = nn.Sequential(
            nn.Linear(384, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        # Duel network - V stream
        self.v1 = nn.Sequential(
            nn.Linear(384, v1_units),
            nn.BatchNorm1d(v1_units),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.v2 = nn.Sequential(
            nn.Linear(v1_units, v2_units),
            nn.BatchNorm1d(v2_units),
            nn.ReLU(),
        )
        self.v3 = nn.Linear(v2_units, 1)

        # Duel network - A stream
        self.a1 = nn.Sequential(
            nn.Linear(384, a1_units),
            nn.BatchNorm1d(a1_units),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.a2 = nn.Sequential(
            nn.Linear(a1_units, a2_units),
            nn.BatchNorm1d(a2_units),
            nn.ReLU(),
        )
        self.a3 = nn.Linear(a2_units, action_size)

    def forward(self, state):
        """Build a Duel network that maps state -> action values."""

        emb = self.emb(state.to(torch.long)).permute(0, 3, 1, 2)

        cc1 = self.cc1(emb)
        cc3 = self.cc3(cc1)

        cc_rows = self.cc_rows(emb)

        cc_cat = torch.cat((cc3.flatten(1), cc_rows.flatten(1)), dim=1)

        fc1 = self.fc1(cc_cat)
        # fc1 = cc_cat

        v1 = self.v1(fc1)
        v2 = self.v2(v1)
        v3 = self.v3(v2)

        a1 = self.a1(fc1)
        a2 = self.a2(a1)
        a3 = self.a3(a2)

        # Use the mean in order to keep the identity of the branches
        # as in formula (9) of https://arxiv.org/pdf/1511.06581.pdf
        q = v3 + (a3 - a3.mean(dim=1, keepdim=True))

        return q
