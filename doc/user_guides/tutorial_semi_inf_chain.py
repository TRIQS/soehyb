
import numpy as np

from mpi4py import MPI

from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, SemiCircular
#from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, inverse, iOmega_n
from triqs.operators import c, c_dag

from triqs_soehyb.solver import Solver, is_root
from triqs_soehyb.triqs_solver import TriqsSolver


if __name__ == '__main__':

    verbose = True
    
    t = 1.0
    half_bandwidth = 2.0

    h_int = 0.0 * c_dag(0,0) * c(0,0)
    
    TS = TriqsSolver(beta=1.0, gf_struct=[(0, 1)], eps=1e-10, w_max=4.0)

    for bidx, delta_tau in TS.Delta_tau:
        delta_w = make_gf_dlr_imfreq(delta_tau)
        delta_w << t**2 * SemiCircular(half_bandwidth)
        delta_tau[:] = make_gf_dlr_imtime(delta_w)

    g_taus = []
    max_order = 4
    orders = range(1, max_order+1)
    for order in orders:
        TS.solve(h_int=h_int, order=order, tol=1e-8)
        g_taus.append(TS.G_tau[0].copy())

    if verbose and is_root():
    
        from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt

        def plot_dlr_imtime(g_tau, label, n_tau=400, marker='x'):

            from triqs.gf import make_gf_imtime

            g_tau_fine = make_gf_imtime(g_tau, n_tau=n_tau)

            color = plt.plot([], [], '-'+marker, label=label)[0].get_color()

            oplotr(g_tau, marker=marker, label=None, color=color)
            oplotr(g_tau_fine, label=None, color=color)
            
        
        subp = [2, 1, 1]

        plt.subplot(*subp); subp[-1] += 1

        delta_tau = TS.Delta_tau[0]
        
        plot_dlr_imtime(delta_tau, 'Bethe', marker='+')

        for order, g_tau in zip(orders, g_taus):
            plot_dlr_imtime(g_tau, label=f'Order {order}')

        plt.legend(loc='best')
        plt.ylabel(r'$G(\tau)$')
        plt.xlabel(r'$\tau$')

        plt.subplot(*subp); subp[-1] += 1

        plt.plot([], [])
        
        for order, g_tau in zip(orders, g_taus):
            diff_tau = delta_tau - g_tau
            diff_tau.data[:] = np.abs(diff_tau.data)
            plot_dlr_imtime(diff_tau, label=f'Order {order}')

        plt.semilogy([], [])
        plt.legend(loc='best')

        plt.ylabel('Abs Error')
        plt.xlabel(r'$\tau$')
            
        plt.tight_layout()
        plt.show()
        

