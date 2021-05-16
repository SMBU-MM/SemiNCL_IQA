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
        return torch.mean(loss)

    def _pcal(self, y1, y1_var, y2, y2_var):
        y_diff = (y1 - y2)
        y_var = y1_var + y2_var 
        p = 0.5 * (1 + torch.erf(y_diff / torch.sqrt(2 * y_var)))
        return p
      
    def forward(self, y1, y1_var, y2, y2_var, g=None):
        n = len(y1)
        p = [self._pcal(y1[i], y1_var[i], y2[i], y2_var[i]) for i in range(n)]
        ################################################################
        # diversity loss
        ################################################################
        comb = [[i, j] for i in range(n) for j in range(i+1, n)]
        div_loss = self._fid(p[comb[0][0]], p[comb[0][1]])
        for i in range(len(comb)-1):
            [m, j] = comb[i+1]
            div_loss += self._fid(p[m], p[j])
        div_loss = div_loss/(n*(n-1))
  
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
            y1_sum, y1_var_sum, y2_sum, y2_var_sum = y1[0].clone(), y1_var[0].clone(), y2[0].clone(), y2_var[0].clone()
            for i in range(n-1):
                y1_sum  += y1[i+1].clone()
                y1_var_sum  += y1_var[i+1].clone()
                y2_sum  += y2[i+1].clone()
                y2_var_sum  += y2_var[i+1].clone()
            p_bar = self._pcal(y1_sum/n, y1_var_sum/(n*n), y2_sum/n, y2_var_sum/(n*n))
            
            e2e_loss = self._fid(p_bar, g)
        # return
        if g==None:
            return div_loss#, div_loss,
        else:
            return e2e_loss, ind_loss, div_loss#, div_loss
