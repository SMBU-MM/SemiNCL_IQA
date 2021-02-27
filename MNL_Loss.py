import torch

eps = 1e-8
class Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))

        return torch.mean(loss)
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
        p = 0.5 * (1 + torch.erf(y_diff / torch.sqrt(2 * y_var.detach())))
        return p

    def forward(self, y1, y1_var, y2, y2_var, g=None):
        n = len(y1)
        p = []
        for i in range(n):
            p.append(self._pcal(y1[i], y1_var[i], y2[i], y2_var[i]))
        ################################################################
        # diversity loss
        ################################################################
        comb = []
        for i in range(n-1):
            for j in range(i+1, n):
                comb.append([i,j])
        for item in enumerate(comb):
            [i, j] = item[1]
            if item[0] == 0:
                div_loss = self._fid(p[i], p[j])
            else:
                div_loss += self._fid(p[i], p[j])
        div_loss = div_loss/(n*(n-1))
        ###############################################################
        # individual empirical loss
        ###############################################################
        if not g==None:
            for i in range(n):
                if i==0:
                    ind_loss = self._fid(p[i], g)
                else:
                    ind_loss += self._fid(p[i], g)
            ind_loss = ind_loss/n
        ###############################################################
        # E2E empirical loss
        ###############################################################
        if not g==None:
            for i in range(n):
                if i == 0:
                    y1_mean = y1[i].clone()
                    y1_var_mean = y1_var[i].clone()
                    y2_mean = y2[i].clone()
                    y2_var_mean = y2_var[i].clone()
                else:
                    y1_mean += y1[i].clone()
                    y1_var_mean += y1_var[i].clone()
                    y2_mean += y2[i].clone()
                    y2_var_mean += y2_var[i].clone()
            p_bar = self._pcal(y1_mean/n, y1_var_mean/n, y2_mean/n, y2_var_mean/n)
            e2e_loss = self._fid(p_bar, g)
        # return
        if g==None:
            return div_loss
        else:
            return e2e_loss, ind_loss, div_loss
