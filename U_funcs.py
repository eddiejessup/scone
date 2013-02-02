import numpy as np

def LJ(r_0, U_0):
    '''
    Lennard-Jones with minimum at (r_0, -U_0).
    '''
    def func(r_sq):
        return U_0 * ((r_0 ** 2 / r_sq) ** 6 - (r_0 ** 2 / r_sq) ** 3)
    return func

def well(r_0, U_0):
    '''
    Potential Well at r with U(r < r_0) = -U_0, U(r > r_0) = 0.
    '''
    def func(r_sq):
        return np.where(r_sq < r_0 ** 2, -U_0, 0.0)
    return func

def inv_sq(k):
    '''
    Inverse-square law, U(r) = -k / r.
    '''
    def func(r_sq):
        return -k / np.sqrt(r_sq)
    return func

def harm_osc(k):
    '''
    Harmonic oscillator, U(r) = k * (r ** 2) / 2.0.
    '''
    def func(r_sq):
        return 0.5 * k * r_sq
    return func

def harm_osc_anis(k):
    '''
    Harmonic oscillator, U(r) = k * (r ** 2) / 2.0.
    '''
    def func(r_sq, theta):
        return (0.5 * k * r_sq) * np.cos(theta)
    return func