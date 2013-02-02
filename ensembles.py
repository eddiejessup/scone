import numpy as np
import utils

k_b = 1.38e-23
h = 3.34e-34

class NVE(object):
    '''
    Microcanonical ensemble of particles with fixed number, volume and
    energy. System is static.
    '''
    def __init__(self, n, d, V, U_func):
        '''
        Initialise a system with n particles in d-dimensional space of
        volume V. Also potential function U_func with parameters U_args.
        '''
        self.n = n
        self.d = d
        self.L = V ** (1.0 / self.d)
        self.U_func = U_func

        self.i = 0
        self.n_moves = 0
        self.r = np.random.uniform(-self.L/2.0, self.L/2.0, (self.n, self.d))
        self.init_arrs()

    def init_arrs(self):
        self.n = len(self.r)
        self.inds = np.arange(self.n)
        self.U = np.zeros(2 * [self.n], dtype=np.float)
        self.U_update = np.ones([self.n], dtype=np.bool)
        self.get_U()

    def get_V(self):
        return self.L ** self.d

    def get_U(self):
        i_update = np.where(self.U_update)[0]
        for i in i_update:
            r_sep_abs = np.abs(self.r[i] - self.r)
            r_sep_abs = np.minimum(r_sep_abs, self.L - r_sep_abs)
            r_sep_sq = utils.vector_mag_sq(r_sep_abs)
            self.U[i, :][self.inds != i] = self.U_func(r_sep_sq[self.inds != i])
            self.U[:, i] = self.U[i, :]
        self.U_update[i_update] = False
        return self.U.sum()

    def iterate_sys(self):
        '''
        Iterate the system's microstate.
        '''
        pass

    def iterate(self):
        '''
        Iterate the environment.
        '''
        self.i += 1
        self.moved = False
        self.iterate_sys()
        self.n_moves += self.moved

class NVT(NVE):
    '''
    Canonical ensemble of particles with fixed number, volume and temperature.
    System may re-arrange particles to minimise its energy.
    '''
    def __init__(self, n, d, V, U_func, T, dr_max):
        '''
        Temperature T with maximum particle position change dr_max.
        '''
        NVE.__init__(self, n, d, V, U_func)
        self.beta = 1.0 / (k_b * T)
        self.dr_max = dr_max

    def iterate_sys(self):
        i = np.random.randint(self.n)
        self.perturb_r(i)

    def displace_r(self, i, dr):
        '''
        Displace a particle at index i by the vector dr.
        '''
        self.r[i] += dr
        self.r[i][self.r[i] > self.L/2.0] -= self.L
        self.r[i][self.r[i] < -self.L/2.0] += self.L
        self.U_update[i] = True

    def perturb_r(self, i):
        '''
        Displace a random particle position with index i and accept or reject
        the move according to the Monte-Carlo acceptance test.
        '''
        U_0 = self.get_U()
        dr = self.dr_max * utils.point_pick_cart(self.d)[0]
        self.displace_r(i, dr)
        dU = self.get_U() - U_0
        P = np.exp(-self.beta * dU)
        if np.minimum(1.0, P) < np.random.uniform():
            self.displace_r(i, -dr)
        else:
            self.moved = True

class NpT(NVT):
    '''
    Isothermal-Isobaric ensemble of particles with fixed number, pressure
    and temperature. System may rearrange particles and change volume to
    minimise its energy.
    '''
    def __init__(self, n, d, V, U_func, T, dr_max, p, dV_max):
        '''
        Pressure p with maximum volume change dV_max.
        '''
        NVT.__init__(self, n, d, V, U_func, T, dr_max)
        self.p = p
        self.dV_max = np.log(1.0 + dV_max)

    def iterate_sys(self):
        i = np.random.randint(self.n + 1)
        if i < self.n:
            self.perturb_r(i)
        else:
            self.perturb_V()

    def displace_V(self, dV_frac):
        '''
        Displace the system volume by fraction dV_frac.
        '''
        dL_frac = dV_frac ** (1.0 / self.d)
        self.L *= dL_frac
        self.r *= dL_frac
        self.U_update[:] = True

    def perturb_V(self):
        '''
        Displace the volume by a random fractional amount and accept or reject
        the move according to the Monte-Carlo acceptance test.
        '''
        U_0 = self.get_U()
        V_0 = self.get_V()
        dV_frac = np.exp(np.random.uniform(-self.dV_max, self.dV_max))
        self.displace_V(dV_frac)
        dU = self.get_U() - U_0
        dV = self.get_V() - V_0
        dlogV = np.log(self.get_V() / V_0)
        P = np.exp(-self.beta * (dU + self.p * dV) + self.n * dlogV)
        if np.minimum(1.0, P) < np.random.uniform():
            self.displace_V(1.0 / dV_frac)
        else:
            self.moved = True

class MVT(NVT):
    '''
    Grand canonical ensemble of particles with fixed chemical potential,
    volume and temperature. System may rearrange, add or remove particles to
    minimise its energy.
    '''
    def __init__(self, n, d, V, U_func, T, dr_max, mu, n_exch):
        '''
        Chemical potential mu with particle exchange probability n_exch.
        '''
        NVT.__init__(self, n, d, V, U_func, T, dr_max)
        self.mu = mu
        self.n_exch = n_exch
        self.lam = (h / np.sqrt(2.0 * np.pi * self.beta)) ** self.d

    def iterate_sys(self):
        i = np.random.randint(self.n + self.n_exch)
        if i < self.n:
            self.perturb_r(i)
        elif np.random.uniform() > 0.5:
            self.perturb_n_up()
        else:
            self.perturb_n_down()

    def displace_n_up(self):
        '''
        Insert a particle at a random position.
        '''
        self.r = np.append(self.r, np.random.uniform(-self.L/2.0, self.L/2.0, (1, self.d)), 0)
        self.init_arrs()

    def perturb_n_up(self):
        '''
        Insert a particle at a random position and accept or reject
        the move according to the Monte-Carlo acceptance test.
        '''
        U_0 = self.get_U()
        self.displace_n_up()
        dU = self.get_U() - U_0
        P = (self.get_V() / (self.lam * self.n)) * np.exp(self.beta * (self.mu - dU))
        if np.minimum(1.0, P) < np.random.uniform():
            self.r = self.r[:-1]
            self.init_arrs()
        else:
            self.moved = True

    def displace_n_down(self):
        '''
        Remove a randomly selected particle.
        '''
        i_del = np.random.randint(self.n)
        r_del = self.r[i_del].copy()
        self.r = np.delete(self.r, [i_del], 0)
        self.init_arrs()
        return r_del

    def perturb_n_down(self):
        '''
        Remove a random particle and accept or reject
        the move according to the Monte-Carlo acceptance test.
        '''
        U_0 = self.get_U()
        r_del = self.displace_n_down()
        dU = self.get_U() - U_0
        P = ((self.lam * (self.n + 1)) / self.get_V()) * np.exp(self.beta * (-self.mu - dU))
        if np.minimum(1.0, P) < np.random.uniform():
            self.r = np.append(self.r, [r_del], 0)
            self.init_arrs()
        else:
            self.moved = True