from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class FFD(nn.Module):
    def __init__(self):
        super(FFD, self).__init__()
        pass

    def forward(self, g_s, g_t):
        return self.mmd_loss(g_s, g_t)

    def mmd_loss(self, f_s, f_t):
        res = (self.poly_kernels(f_t, f_t).mean().detach() + self.poly_kernels(f_s, f_s).mean()
                 - 2 * self.poly_kernels(f_s, f_t).mean())
        return res

    def poly_kernels(self, a, b):
        a = a.transpose(1, 2)
        res = torch.bmm(a,b)
        res = res.pow(2)
        return res

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res


if __name__ == '__main__':
    data1 = torch.tensor(np.random.normal(loc=0, scale=10, size=(1, 64, 1024)))
    data2 = torch.tensor(np.random.normal(loc=10, scale=5, size=(1, 64, 1024)))
    criterion_kd = FFD()
    loss = criterion_kd(data1, data2)
    print((loss))
