import numpy as np
import utils
import ensembles

class NVE_polar(ensembles.NVE):
    '''
    Microcanonical ensemble of polar particles with fixed number, volume and
    energy. System is static.
    '''
    def __init__(self, **kwargs):
        ensembles.NVE.__init__(self, **kwargs)
        self.th = np.random.uniform(-np.pi, np.pi, self.n)

    def get_U(self):
        for i in np.where(self.U_changed)[0]:
            r_sep_sq = self.get_r_sep_sq(i)
            dtheta = self.th[i] - self.th
            # Potential of me due to everyone
            self.U[i, :][self.inds != i] = self.U_func(r_sep_sq[self.inds != i], dtheta[self.inds != i])
            # Potential of everyone else due to me
            self.U[:, i][self.inds != i] = self.U_func(r_sep_sq[self.inds != i], -dtheta[self.inds != i])
        self.U_changed[:] = False
        return self.U.sum()

class NVT_polar(ensembles.NVT, NVE_polar):
    def __init__(self, dth_max, **kwargs):
        self.dth_max = dth_max
        ensembles.NVT.__init__(self, **kwargs)
        NVE_polar.__init__(self, **kwargs)

    def perturb_micro(self, i):
        if np.random.uniform() > 0.5:
            self.perturb_r(i)
        else:
            self.perturb_th(i)

    def displace_th(self, i, dth):
        self.th[i] += dth
        self.U_changed[i] = True

    def perturb_th(self, i):
        U_0 = self.get_U()
        dth = np.random.uniform(-self.dth_max, self.dth_max)
        self.displace_th(i, dth)
        dU = self.get_U() - U_0
        P = np.exp(-self.beta * dU)
        if np.minimum(1.0, P) < np.random.uniform():
            self.displace_th(i, -dth)
        else:
            self.moved = True

class NpT_polar(ensembles.NpT, NVT_polar):
    def __init__(self, **kwargs):
        ensembles.NpT.__init__(self, **kwargs)
        NVT_polar.__init__(self, **kwargs)

class MVT_polar(ensembles.MVT, NVT_polar):
    def __init__(self, **kwargs):
        ensembles.MVT.__init__(self, **kwargs)
        NVT_polar.__init__(self, **kwargs)

    def get_dat_new(self):
        return MVT.get_dat_new(self) + (np.random.uniform(np.pi, np.pi),)

    def displace_n_up(self, r, th):
        self.th = np.append(self.th, th)
        return MVT.displace_n_up(self, r)

    def displace_n_down(self, i):
        th_del = self.th[i]
        self.th = np.delete(self.th, [i])
        return MVT.displace_n_down(self, i) + (th_del,)