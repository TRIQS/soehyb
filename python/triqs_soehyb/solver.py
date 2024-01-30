
import numpy as np

from mpi4py import MPI as mpi

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization


from .pycppdlr import build_dlr_rf
from .pycppdlr import ImTimeOps

from .impurity import Fastdiagram
from .diag import all_connected_pairings


def is_root():
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    return rank == 0


def scatter_array_over_ranks(arr):
    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    arr_rank = np.array_split(np.array(arr), size, axis=0)[rank]
    return arr_rank


def Sigma_calc_loop(fd, G_iaa, order, verbose=False):

    assert( order >= 1 )
    assert( type(fd) == Fastdiagram )

    Sigma_iaa = np.zeros_like(G_iaa)

    for order in range(1, order+1):
        n_diags = fd.number_of_diagrams(order)
        if verbose and is_root():
            print(f"order = {order}")
            print(f'n_diags = {n_diags}')
        for sign, diag in all_connected_pairings(order):
            if verbose and is_root():
                print(sign, diag)
            diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
            diag_idx_vec = np.arange(n_diags, dtype=np.int32)
            diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
            Sigma_iaa -= sign * fd.Sigma_calc_group(G_iaa, diag_vec, diag_idx_vec)

    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, Sigma_iaa)
            
    return Sigma_iaa


def G_calc_loop(fd, G_iaa, order, n_orb, verbose=False):

    assert( order >= 1 )
    assert( type(fd) == Fastdiagram )

    n_dlr = G_iaa.shape[0]
    g_iaa = np.zeros((n_dlr, n_orb, n_orb), dtype=complex)

    for order in range(1, order+1):
        n_diags = fd.number_of_diagrams(order)
        if verbose and is_root():
            print(f"order = {order}")
            print(f'n_diags = {n_diags}')
        for sign, diag in all_connected_pairings(order):
            if verbose and is_root():
                print(sign, diag)
            diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
            diag_idx_vec = np.arange(n_diags, dtype=np.int32)
            diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
            g_iaa -= sign * fd.G_calc_group(G_iaa, diag_vec, diag_idx_vec)

    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, g_iaa)
            
    return g_iaa


class Solver(object):

    
    def __init__(self, beta, lamb, eps, H_loc, fundamental_operators):

        self.lamb = lamb
        self.eps = eps
        
        self.dlr_rf = build_dlr_rf(lamb, eps)
        self.ito = ImTimeOps(lamb, self.dlr_rf)
    
        self.beta = beta
        self.H_loc = H_loc
        self.fundamental_operators = fundamental_operators

        self.ed = TriqsExactDiagonalization(H_loc, fundamental_operators, beta)

        mat_c_dag = np.array(self.ed.rep.sparse_operators.c_dag[0].todense()) 
        mat_c = mat_c_dag.T.conj()
        
        self.F = np.array([mat_c])
        self.F_dag = np.array([mat_c_dag])
    
        self.fd = Fastdiagram(beta, lamb, eps, self.F, self.F_dag)
        self.tau_i = self.fd.get_it_actual()
        
        self.H_mat = np.array(self.ed.ed.H.todense())
    
        self.G0_iaa = self.fd.free_greens_ppsc(beta, self.H_mat)
        self.G_iaa = self.G0_iaa.copy()

        self.eta = 0.


    def set_hybridization(self, Delta_iaa,poledlrflag=True,eps=1e-8):

        if poledlrflag==True:
            self.fd.hyb_init(Delta_iaa)
            self.fd.hyb_decomposition()
        else:
            from triqs_soehyb.ac_pes import polefitting
            self.fd.hyb_init(Delta_iaa,poledlrflag)
            weights, pol, error = polefitting(self.fd.Deltaiw, 1j*self.fd.dlr_if,eps= eps)
            weights_reflect, pol_reflect, error_reflect = polefitting(self.fd.Deltaiw_reflect, 1j*self.fd.dlr_if,eps= eps)
            self.fd.copy_aaa_result(pol, weights,pol_reflect,weights_reflect)
            self.fd.hyb_decomposition(poledlrflag)


    def solve(self, order, tol=1e-9, maxiter=10, mix=1.0, verbose=True):

        for iter in range(maxiter):

            Sigma_iaa = Sigma_calc_loop(self.fd, self.G_iaa, order, verbose=verbose)        
            G_iaa_new = self.fd.time_ordered_dyson(self.beta, self.H_mat, self.eta, Sigma_iaa)

            Z = self.fd.partition_function(G_iaa_new)
            deta = np.log(Z) / self.beta
            G_iaa_new[:] *= np.exp(-self.tau_i * deta)[:, None, None]
            self.eta += deta

            diff = np.max(np.abs(self.G_iaa - G_iaa_new))

            self.G_iaa = mix*G_iaa_new + (1-mix)*self.G_iaa 

            if is_root():
                print(f'PPSC: iter = {iter:3d} diff = {diff:2.2E}')
            if diff < tol: break


    def calc_spgf(self, order, verbose=True):
        
        n_orb = self.F.shape[0]
        g_iaa = G_calc_loop(self.fd, self.G_iaa, order, n_orb, verbose=verbose)
        
        return g_iaa
        
