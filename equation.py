import numpy as np
import torch
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, dim, total_time, num_time_interval):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_th(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_th(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    def h_th(self, x, y):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    def get_x_init(self, shape_val):
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")


class AllenCahn(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(AllenCahn, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t

        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def get_x_init(self, shape_val):
        return torch.zeros(size=[shape_val, self.dim])

    def f_th(self, t, x, y, z):
        return y - torch.pow(y, 3)

    def g_th(self, t, x):
        return 0.5 / (1 + 0.2 * torch.sum(x**2, dim=1, keepdim=True))

    def h_th(self, x, y):
        return 0


class LiMan(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(LiMan, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self.sigma = 1
        self._rho = 3

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        dw_sample = dw_sample.reshape((num_sample, self._dim, self._num_time_interval))
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def get_x_init(self, shape_val):
        return np.zeros(self._dim)

    def f_th(self, t, x, y, z):
        return - torch.arctan(torch.mean(x) * torch.ones(x.shape))

    def g_th(self, t, x):
        return torch.arctan(x)

    def h_th(self, x, y):
        return - self._rho * y


class LiMan2(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(LiMan2, self).__init__(dim, total_time, num_time_interval)
        self.x_init = 2
        self.sigma = 1
        self._rho = 0.1
        self.a = 0.25

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        dw_sample = dw_sample.reshape((num_sample, self._dim, self._num_time_interval))
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def get_x_init(self, shape_val):
        return torch.ones(size=[shape_val, self.dim]) * self.x_init

    def f_th(self, t, x, y, z):
        return self.a * y

    def g_th(self, t, x):
        return x

    def h_th(self,x, y):
        return - self._rho * torch.mean(y) * torch.ones(y.shape)


class LiMan3(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(LiMan3, self).__init__(dim, total_time, num_time_interval)
        self.x_init = 2.0
        self.sigma = 1.5
        self.B = 0.2
        self.A = 0.1
        self.R = 5.0
        self.M = 500.0
        self.Q = 0.1

    def get_x_init(self, shape_val):
        return torch.normal(0.3, 1, size=[shape_val, self.dim])

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        dw_sample = dw_sample.reshape((num_sample, self._dim, self._num_time_interval))
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        return 2 * self.Q * (x - torch.mean(x) * torch.ones(x.shape)) + self.A * y

    def g_th(self, t, x):
        return torch.where(x >= 0, 2 * (x - 10) * self.M, 2 * (x + 10) * self.M)

    def h_th(self, x, y):
        return self.A * x - 0.5 * pow(self.B, 2) / self.R * y
