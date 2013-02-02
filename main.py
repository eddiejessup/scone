import csv
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils
import ensembles
import U_funcs

every = 500
random_seed = 100
i_max = 400000

# Potential parameters
#U_func = U_funcs.well(r_0=0.05, U_0=-100000)
#U_func = U_funcs.LJ(r_0=0.05, U_0=10.0)
#U_func = U_funcs.inv_sq(k=-1.0)
U_func = U_funcs.harm_osc_anis(k=01.0)

# NVE parameters
d = 2
n = 500
V = 1.0

# NVT parameters
T = 300.0
dr_max = 1e-2

# NPT parameters
p = 100.0
dV_max = 1e-1

# MVT parameters
mu = 0.1
n_exch = 5

def main():
    np.random.seed(random_seed)
#    system = ensembles.NVE(n, d, V, U_func)
#    system = ensembles.NVT(n, d, V, U_func, T, dr_max)
#    system = ensembles.NpT(n, d, V, U_func, T, dr_max, p, dV_max)
#    system = ensembles.MVT(n, d, V, U_func, T, dr_max, mu, n_exch)
    system = ensembles.NVE_polar(n, d, V, U_func)

    # Output
    f_output = open('outputn.dat', 'w')
    output = csv.writer(f_output, delimiter=' ')
    output.writerow(['i', 'U', 'dyn'])
    # Plotting
    utils.makedirs_soft('p')
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
            # Output
            print system.i, system.get_U()
            moves.append(system.n_moves)
            output.writerow([system.i, system.get_U(), (moves[-1] - moves[-2]) / float(every)])
            f_output.flush()
            # Plotting
            if system.d == 2:
                plot.set_offsets(system.r)
            elif system.d == 3:
                plot._offsets3d = (system.r[:, 0], system.r[:, 1], system.r[:, 2])
                ax.set_zlim([-system.L/2.0, system.L/2.0])
            ax.set_xlim([-system.L/2.0, system.L/2.0])
            ax.set_ylim([-system.L/2.0, system.L/2.0])
            fig.canvas.draw()
            fig.savefig('p/%010i.png' % system.i)
        system.iterate()


if __name__ == '__main__':
    main()