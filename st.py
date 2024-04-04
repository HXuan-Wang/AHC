import torch 
import torch.nn.functional as F
import torch.nn as nn
def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=2.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

class CMMDLoss(nn.Module):
    def __init__(self):
        super(CMMDLoss, self).__init__()
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