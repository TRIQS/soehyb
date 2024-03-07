import numpy as np
import time
import matplotlib.pyplot as plt
from triqs_soehyb.ac_pes import *
from triqs_soehyb.diag import *
from mpi4py import MPI as mpi
from triqs.operators import c, c_dag,n,Operator
from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization
from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori
from itertools import product
import itertools
from triqs.gf import GfImTime
from triqs_soehyb.solver import *
def construct_impurity_Hamiltonian(norbI, U, J, mu):

        
        spin_names = ('up','do')
        orb_names = list(range(norbI))
    
        fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]
    
        KanMat1, KanMat2 = U_matrix_kanamori(norbI, U, J)
        H = h_int_kanamori(spin_names, norbI, KanMat1, KanMat2, J, True)

        N_up = n('up',0) + n('up',1)
        N_do = n('do',0) + n('do',1)
        H -= mu *(N_up + N_do)
        return H,  fundamental_operators
def construct_delta_iaa(fd, ek,beta):
    Hbath = np.diag([ek, -ek])
    g_p = fd.free_greens(beta, np.array([[ek]]))
   
    g_m = fd.free_greens(beta, np.array([[-ek]]))
    g = g_p + g_m
    
    T = np.array([
        [1, r0,0,0],
        [r0, 1,0,0],
        [0,0,1,r0],
        [0,0,r0,1]])

    return T[None, ...] * g
if __name__ == '__main__':

    beta, eps, mix, ppsc_tol, ppsc_maxiter =8.0, 1e-10, 0.0 ,1e-6,20
    lamb = 10 * beta

      
    U, J, ek, a, r0, dmu, norbI = 2.0, 0.2, 2.3, 1.0, 1.0, -1.5, 2
    mu = (3*U-5*J)/2.0 + dmu
    ntau = 1000

    #construct impurity Hamiltonian
    H_loc, fundamental_operators = construct_impurity_Hamiltonian(norbI, U, J, mu)

    #initialize solver
    Impurity = Solver(beta, lamb, eps, H_loc, fundamental_operators)
    
    # Bath delta_iaa
    delta_iaa = construct_delta_iaa(Impurity.fd, ek, beta)
    #polefitting
    Impurity.set_hybridization(delta_iaa,poledlrflag=False,delta_diff = 1.0,fittingeps = 1e-6,verbose=True,Hermitian=True)
   
    #tau grid that on which we will evaluate green's functions
    tau_grid = np.linspace(0,1,ntau)
    n = delta_iaa.shape[1]
    g_mesh_all = np.zeros((5,tau_grid.shape[0],n,n),complex)

    #itops tools
    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)

    verbose=True
    #ppsc iterations
    for max_order in [1,2,3]: 
        Impurity.solve(max_order,  ppsc_maxiter=ppsc_maxiter,ppsc_tol = ppsc_tol, update_eta_exact = False , verbose=verbose)          
        g_iaa = Impurity.calc_spgf(max_order, verbose=verbose)
        g_xaa = ito.vals2coefs(g_iaa)
        
        def interp(g_xaa, tau_j):
            eval = lambda t : ito.coefs2eval(g_xaa, t)
            return np.vectorize(eval, signature='()->(m,m)')(tau_j)
        g_on_mesh = interp(g_xaa, tau_grid)
        g_mesh_all[max_order,:,:,:] = g_on_mesh

        if is_root():
            plt.plot(tau_grid, g_on_mesh[:,0,0])
    if is_root():
        
        filename = "result_twoband_discrete_maxorder="+str(max_order)+"_beta="+str(beta)+".npy"
        np.save(filename,g_mesh_all)
        plt.ylim([-1,0])
        plt.legend(["NCA","OCA","TCA","4th"])
        plt.title("beta = "+str(beta))
        plt.show()