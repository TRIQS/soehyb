import numpy as np
import matplotlib.pyplot as plt
from triqs_soehyb.ac_pes import *
from triqs_soehyb.diag import *
from mpi4py import MPI as mpi
import itertools
from triqs.operators import c, c_dag,n
from triqs.gf import GfImTime
from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization
from itertools import product
from triqs_soehyb.solver import *
import matplotlib.pyplot as plt
from triqs_soehyb.solver import *

def calc_two_band_spinless_ed(
        ntau=500,
        beta = 4.0,
        U=4.0,
        v=1.5,
        t=1.0,
        t1=1.5,
        ek=0.0,
        mu=0.0,
        eps=None, ppsc_tol=None, lamb=None,
        return_ed=False,
        ): 
    
    n_i = [ c_dag(0,i) * c(0,i) for i in range(2)]
    n, d = np.sum(n_i), np.product(n_i)
    
    # Impurity
    H = U * d - mu * n - v * (c_dag(0,0) * c(0,1) + c_dag(0,1) * c(0,0))

    # bath
    n_b = [ c_dag(0,i) * c(0,i) for i in range(2,6)]

    nbath = np.sum(n_b) 
    
    H += ek * nbath - t * (
        c_dag(0,0) * c(0,2) + c_dag(0,2) * c(0,0) + \
        c_dag(0,0) * c(0,3) + c_dag(0,3) * c(0,0) + \
        c_dag(0,1) * c(0,4) + c_dag(0,4) * c(0,1) + \
        c_dag(0,1) * c(0,5) + c_dag(0,5) * c(0,1))
    
    H += - t1 * (
        c_dag(0,2) * c(0,4) + c_dag(0,4) * c(0,2) + \
        c_dag(0,3) * c(0,5) + c_dag(0,5) * c(0,3))
    
    fundamental_operators = [ c(0,i) for i in range(6) ]
    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

    if return_ed: return ed
    
    g_tau = GfImTime(name=r'$g$', beta=beta, statistic='Fermion', n_points=ntau, target_shape=(2,2))

    for i, j in itertools.product(range(2), repeat=2):
        ed.set_g2_tau(g_tau[i,j], c(0,i), c_dag(0,j) )

    return g_tau

if __name__ == '__main__':
    
    beta = 2.0
    t, U, v, t1, ek, mu= 1.0, 4.0, 1.5, 1.5, 0.0, 0.0
    ntau = 1000
    gf_true = calc_two_band_spinless_ed(ntau,beta,U,v,t,t1,ek,mu)
    


    n_i = [ c_dag(0,i) * c(0,i) for i in range(2)]
    n, d = np.sum(n_i), np.product(n_i)
    
    # Impurity
    H_loc = U * d - mu * n - v * (c_dag(0,0) * c(0,1) + c_dag(0,1) * c(0,0))
    fundamental_operators = [ c(0,i) for i in range(2) ]

    lamb, eps = beta*100, 1.0e-12
    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)
    #construct solver
    Impurity = Solver(beta, lamb, eps, H_loc, fundamental_operators)

    #hybridization function
    Hbath=np.array([[ek,-t1],[-t1,ek]])
    delta_iaa = 2* Impurity.fd.free_greens(beta,Hbath)* (t**2) 

    #hybridization expansion
    Impurity.set_hybridization(delta_iaa,poledlrflag=False,delta_diff = 1.0,fittingeps = 1e-6,verbose=True,Hermitian=True)

   
    #obtain actual imaginary time nodes on [0,beta]
    tau_grid = np.linspace(0,1,ntau)
    g_mesh_all = np.zeros((6,tau_grid.shape[0],delta_iaa.shape[1],delta_iaa.shape[1]),complex)

    ppsc_maxiter, ppsc_tol =10, 1e-10
    verbose=True
    error = []
    for max_order in [1,2,3]:
        Impurity.solve(max_order,  ppsc_maxiter=ppsc_maxiter,ppsc_tol = ppsc_tol, update_eta_exact = False , verbose=verbose)          
        g_iaa = Impurity.calc_spgf(max_order, verbose=verbose)

        g_xaa = ito.vals2coefs(g_iaa)
        
        def interp(g_xaa, tau_j):
            eval = lambda t : ito.coefs2eval(g_xaa, t)
            return np.vectorize(eval, signature='()->(m,m)')(tau_j)
        g_on_mesh = interp(g_xaa, tau_grid)
        plt.plot(tau_grid,g_on_mesh[:,0,0].real)
        plt.ylim([-1,0])
        error.append(np.max(np.abs(gf_true.data - g_on_mesh)))
        g_mesh_all[max_order, :, :, :] = g_on_mesh
        
    plt.legend(["1","2","3"])
    plt.show()
    filename = "result_twoband_maxorder="+str(max_order)+"_beta="+str(beta)+".npy"
    np.save(filename,g_mesh_all)
    breakpoint()