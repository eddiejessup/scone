from __future__ import print_function
import csv
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils
import potentials
import ensembles
import ensembles_polar

every = 200
random_seed = 12
i_max = 400000

args = {}
# Potential
args['U_func'] = potentials.LJ(r_0=0.05, U_0=1.0)
args['U_func'] = potentials.anis_wrap(args['U_func'])

# NVE
args['d'] = 2
args['n'] = 400
args['V'] = 1.0

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
args['dth_max'] = 0.1

def main():
    np.random.seed(random_seed)
#    system = ensembles.NVE(**args)
#    system = ensembles.NVT(**args)
#    system = ensembles.NpT(**args)
#    system = ensembles.MVT(**args)

#    system = ensembles_polar.NVE_polar(**args)
#    system = ensembles_polar.NVT_polar(**args)
    system = ensembles_polar.NpT_polar(**args)
#    system = ensembles_polar.MVT_polar(**args)
    pp.show()
    pp.ion()

    # Output
    f_output = open('Data/log.dat', 'w')
    output = csv.writer(f_output, delimiter=' ')
    output.writerow(['i', 'U', 'dyn'])

    # Plotting
    utils.makedirs_soft('Data/p')
    fig = pp.figure()
    if system.d == 2:
        ax = fig.add_subplot(111)
        plot = ax.scatter([], [])
    elif system.d == 3:
        ax = fig.add_subplot(111, projection='3d')
        plot = ax.scatter([], [], [])
    ax.set_aspect('equal')
    fig.show()

    moves = [system.n_moves]

    while system.i <= i_max:
        if not system.i % every:

#            pp.quiver(system.r[:, 0], system.r[:, 1], np.cos(system.th), np.sin(system.th), pivot='middle', headwidth=0)
#            pp.xlim([-system.L/2.0, system.L/2.0])
#            pp.ylim([-system.L/2.0, system.L/2.0])
#            pp.gca().set_aspect('equal')
#            pp.draw()
#            pp.cla()

            # Output
            moves.append(system.n_moves)
            output.writerow([system.i, system.get_U(), (moves[-1] - moves[-2]) / float(every)])
            f_output.flush()

#            thetas = system.th.copy()
#            thetas %= np.pi
#            print np.std(thetas)

            # Plotting
            if system.d == 2:
                plot.set_offsets(system.r)
            elif system.d == 3:
                plot._offsets3d = (system.r[:, 0], system.r[:, 1], system.r[:, 2])
                ax.set_zlim([-system.L/2.0, system.L/2.0])
            ax.set_xlim([-system.L/2.0, system.L/2.0])
            ax.set_ylim([-system.L/2.0, system.L/2.0])
            fig.canvas.draw()
            fig.savefig('Data/p/%010i.png' % system.i)

        system.iterate()

if __name__ == '__main__':
    main()