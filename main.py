from __future__ import print_function
import csv
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils
import potentials
import ensembles
import ensembles_polar

every = 500

i_max = 400000

random_seed = 12
n = 500

def get_inits(n, d, V=1.0, seed=None):
    np.random.seed(seed)
    L = V ** (1.0 / d)
    r = np.random.uniform(-L/2.0, L/2.0, (n, d))
    th = np.random.uniform(-np.pi, np.pi, len(r))
    return r, th

args = {}
# Potential
args['U_func'] = potentials.LJ(r_0=0.05, U_0=1.0)
args['U_func'] = potentials.anis_wrap(args['U_func'], potentials.polar_rose_sq(2))

# NVE
args['d'] = 2
args['V'] = 1.0
args['r'], args['th'] = get_inits(n, args['d'], args['V'], random_seed)

# NVT
args['T'] = 300
args['dr_max'] = 5e-2

# NpT
args['p'] = 10000.0
args['dV_max'] = 1e-2

# MVT
args['mu'] = 0.1
args['n_exch'] = 5

# Polar
args['dth_max'] = 0.2

def main():
#    system = ensembles.NVE(**args)
#    system = ensembles.NVT(**args)
#    system = ensembles.NpT(**args)
#    system = ensembles.MVT(**args)

#    system = ensembles_polar.NVE(**args)
    system = ensembles_polar.NVT(**args)
#    system = ensembles_polar.NpT(**args)
#    system = ensembles_polar.MVT(**args)

    polar = isinstance(system, ensembles_polar.NVE)

    # Output
    utils.makedirs_soft('Data')
    f_output = open('Data/log.dat', 'w')
    output = csv.writer(f_output, delimiter=' ')
    headers = ['i', 'U', 'dyn']
    if polar: headers.append('th_std')
    output.writerow(headers)

    # Plotting
    utils.makedirs_soft('Data/p')
    fig = pp.figure()
    if system.d == 2:
        ax = fig.add_subplot(111)
        if polar:
            plot = ax.quiver(system.r[:, 0], system.r[:, 1], np.cos(system.th), np.sin(system.th), pivot='middle', headwidth=0)
        else:
            plot = ax.scatter(system.r[:, 0], system.r[:, 1])
    elif system.d == 3:
        ax = fig.add_subplot(111, projection='3d')
        plot = ax.scatter([], [], [])
        ax.set_zlim([-system.L/2.0, system.L/2.0])
    ax.set_xlim([-system.L/2.0, system.L/2.0])
    ax.set_ylim([-system.L/2.0, system.L/2.0])
    ax.set_aspect('equal')
    fig.show()

    moves = [system.n_moves]

    while system.i <= i_max:
        if not system.i % every:

            # Output
            moves.append(system.n_moves)
            data = [system.i, system.get_U(), (moves[-1] - moves[-2]) / float(every)]
            if polar: data.append(np.std(system.th % np.pi))
            print(data)
            output.writerow(data)
            f_output.flush()

            # Plotting
            if system.d == 2:
                plot.set_offsets(system.r)
                plot.set_UVC(np.cos(system.th), np.sin(system.th))
            elif system.d == 3:
                plot._offsets3d = (system.r[:, 0], system.r[:, 1], system.r[:, 2])
            fig.canvas.draw()
    #        fig.savefig('Data/p/%010i.png' % system.i)

        system.iterate()

if __name__ == '__main__':
    main()