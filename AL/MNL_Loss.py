import torch
import torch.nn.functional as F 

eps = 1e-8

class Ncl_loss(torch.nn.Module):
    def __init__(self):
        super(Ncl_loss, self).__init__()

    def _fid(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))
        return loss

    def _pcal(self, y1, y1_var, y2, y2_var):
        y_diff = (y1 - y2)
        y_var = y1_var + y2_var 
        p = 0.5 * (1 + torch.erf(y_diff / torch.sqrt(2 * y_var)))
        return p
      
    def forward(self, y1, y1_var, y2, y2_var, y1_ens, y1_ens_var, y2_ens, y2_ens_var, g=None):
        n = len(y1)
        p = [self._pcal(y1[i], y1_var[i], y2[i], y2_var[i]) for i in range(n)]
        p_ens = self._pcal(y1_ens, y1_ens_var, y2_ens, y2_ens_var)
        div_loss = self._fid(p[0], p_ens)
        for i in range(n-1):
            div_loss += self._fid(p[i+1], p_ens)
        div_loss = div_loss/n
        
        if not g==None:
            ###############################################################
            # individual empirical loss
            ###############################################################
            ind_loss_l = [self._fid(p[i], g) for i in range(n)]
            ind_loss = ind_loss_l[0]
            for i in range(n-1):
                ind_loss += ind_loss_l[i+1]
            ind_loss= ind_loss/n
            ###############################################################
            # E2E empirical loss
            ###############################################################
            e2e_loss = self._fid(p_ens, g)
        # return
        if g==None:
            return torch.mean(div_loss)#, #div_loss
        else:
            return torch.mean(e2e_loss), torch.mean(ind_loss), torch.mean(div_loss)#, div_loss
