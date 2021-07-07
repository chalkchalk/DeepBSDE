import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

TH_DTYPE = torch.float32

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50


class Dense(nn.Module):

    def __init__(self,cin,cout, batch_norm=True, activate=True):
        super(Dense, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.activate = activate
        if batch_norm:
            self.bn = nn.BatchNorm1d(cout,eps=EPSILON, momentum=MOMENTUM)
        else:
            self.bn = None
        nn.init.normal_(self.linear.weight,std=5.0/np.sqrt(cin+cout))

    def forward(self,x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate:
            x = F.relu(x)
        return x


class Subnetwork(nn.Module):

    def __init__(self, config):
        super(Subnetwork, self).__init__()
        self._config = config
        self.bn = nn.BatchNorm1d(config.dim, eps=EPSILON, momentum=MOMENTUM)
        self.layers = [Dense(config.num_hiddens[i-1], config.num_hiddens[i]) for i in range(1, len(config.num_hiddens)-1)]
        self.layers += [Dense(config.num_hiddens[-2], config.num_hiddens[-1], activate=False)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.bn(x)
        x = self.layers(x)
        return x


class FeedForwardModel(nn.Module):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        super(FeedForwardModel, self).__init__()
        self._config = config
        self._bsde = bsde

        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time

        self._y_init = Parameter(torch.Tensor([1]))
        self._y_init.data.uniform_(self._config.y_init_range[0], self._config.y_init_range[1])
        self._subnetworkList = nn.ModuleList([Subnetwork(config) for _ in range(self._num_time_interval-1)])

    def forward(self, xs, dw):

        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        z_init = torch.zeros([1, self._dim]).uniform_(-.1, .1).to(TH_DTYPE)
        all_one_vec = torch.ones((dw.shape[0], 1), dtype=TH_DTYPE)
        y = all_one_vec * self._y_init
        x = self._bsde.get_x_init(xs.shape[0])
        # x = torch.ones([xs.shape[0], self._dim]) * self._bsde.x_init
        z = torch.matmul(all_one_vec, z_init)
        for t in range(0, self._num_time_interval-1):
            y = y - self._bsde.delta_t * (
                self._bsde.f_th(time_stamp[t], x, y, z)
                )
            y = y + torch.sum(z * dw[:, :, t], dim=1, keepdim=True) #torch.Size([64, 1])
            x = x + self._bsde.delta_t * self._bsde.h_th(x, y) + self._bsde.sigma * dw[:, :, t]
            z = self._subnetworkList[t](x) / self._dim
        y = y - self._bsde.delta_t * self._bsde.f_th(
                    time_stamp[-1], x, y, z
                    ) + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)
        x = x + self._bsde.delta_t * self._bsde.h_th(x, y) + self._bsde.sigma * dw[:, :, -1]
        X_N = self._bsde.g_th(self._total_time, x)
        # print("===")
        # print(torch.mean(x))
        #print(torch.mean(X_N))
        delta = y - X_N
        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, delta**2,
                                                    2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2))
        return loss, self._y_init




