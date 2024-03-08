
import time

import numpy as np
from scipy.optimize import root_scalar

from mpi4py import MPI as mpi

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from .pycppdlr import build_dlr_rf
from .pycppdlr import ImTimeOps

from .ac_pes import polefitting
from .impurity import Fastdiagram
from .diag import all_connected_pairings

def kernel(tau, omega):
    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

    return kernel

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


def Sigma_calc_loop(fd, G_iaa,max_order,verbose=True):

    assert( max_order >= 1 )
    assert( type(fd) == Fastdiagram )
    
    if verbose:
        start_time = time.time()    

    Sigma_t = np.zeros((G_iaa.shape[0], G_iaa.shape[1], G_iaa.shape[2]), dtype=complex)
    
    for ord in range(1, max_order+1):
        n_diags = fd.number_of_diagrams(ord)
        if is_root() and verbose:
            print(f"order = {ord}")
            print(f'n_diags = {n_diags}')
        for sign, diag in all_connected_pairings(ord):
            if  is_root() and verbose:
                print(sign, diag)
            diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
            diag_idx_vec = np.arange(n_diags, dtype=np.int32)
            diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
            Sigma_t += pow(-1,ord)* sign * fd.Sigma_calc_group(G_iaa, diag_vec, diag_idx_vec)

    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, Sigma_t)

    if is_root() and verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time spent is ",elapsed_time)

    return Sigma_t


def G_calc_loop(fd, G_iaa, max_order,n_g,verbose=True):

    if verbose:
        start_time = time.time()
    
    g_iaa = np.zeros((G_iaa.shape[0], n_g, n_g), dtype=complex)

    for ord in range(1, max_order+1):
        n_diags = fd.number_of_diagrams(ord)
        if is_root() and verbose:
            print(f"order = {ord}")
            print(f'n_diags = {n_diags}')
        for sign, diag in all_connected_pairings(ord):
            if  is_root():
                print(sign, diag)
            diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
            diag_idx_vec = np.arange(n_diags, dtype=np.int32)
            diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
            g_iaa += pow(-1,ord)* sign * fd.G_calc_group(G_iaa, diag_vec, diag_idx_vec)

    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, g_iaa) 

    if is_root() and verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time spent is ",elapsed_time)
        
    return g_iaa


def g_iaa_reconstruct(poles,weights,tau_i):
    G = np.zeros((tau_i.shape[0],weights.shape[1],weights.shape[1]), dtype = np.complex128)
    
    for n in range(poles.shape[0]): 
        for i in range(tau_i.shape[0]):
            # print(i)
            G[i,:,:] = G[i,:,:] + kernel(tau_i[i:i+1],poles[n:n+1])[0,0]*weights[n,:,:]
    return G


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

        self.F_dag = np.array([np.array(self.ed.rep.sparse_operators.c_dag[idx].todense()) for idx in range(len(fundamental_operators))])
        self.F = np.array([np.array(
            self.ed.rep.sparse_operators.c_dag[idx].T.conj().todense())
            for idx in range(len(fundamental_operators)) ])

        
        self.H_mat = np.array(self.ed.ed.H.todense())

        self.fd = Fastdiagram(beta, lamb, eps, self.F, self.F_dag)
        self.tau_i = self.fd.get_it_actual().real
        self.G0_iaa = self.fd.free_greens_ppsc(beta, self.H_mat)
        self.G_iaa = self.G0_iaa.copy()

        self.eta = 0.

        
    def set_hybridization(self, delta_iaa,
                          poledlrflag=True, delta_diff=1.0, fittingeps=2e-6,
                          Hermitian=False, verbose=True):
        
        """ TODO: Explain interplay of all the flags

        Set the hybridization function of the expansion.

        Parameters
        ----------

        delta_iaa : (n, m, m) ndarray
            Hybridization function on imaginary time DLR nodes.

        poledlrflag : bool, optional
            What does this flag do?
            Use the full DLR representation to represent the hybridization function.
            Default `True`
        
            (How about doint the inverse logic? and replace `poledlrflag` with flag called `compress=False`)
            I think this would be more intuitive for the user to understand.

        delta_diff : float, optional
            current difference of hybridization functions in DMFT iterations. 
            The pole fitting tolerance will be enforced to be smaller than delta_diff/1000 to ensure the pole fitting error will not affect DMFT iterations.
            In one-shot impurity solvers, one can neglect this argument. 
            Default 1.0

        fittingeps : float, optional
            The pole fitting error tolerance
            Default 2e-6

        Hermitian : bool, optional
            Choice of whether to enforce the weight matrices to be Hermitian or not. 
            Default `False`

        verbose : bool, optional
            Enable (more) verbose printouts of the method.
            Defailt `True`
        
        """

        
        if poledlrflag == True:
           
            self.fd.hyb_init(delta_iaa, poledlrflag=True)
            self.fd.hyb_decomposition(poledlrflag=True, eps=fittingeps/10)
            
        else:
            #decomposition and reflection of Delta(t) using aaa poles
            
            self.fd.hyb_init(delta_iaa, poledlrflag)
            epstol = min(fittingeps, delta_diff/1000)
            Npmax = len(self.fd.dlr_if)-1
            weights, pol, error = polefitting(self.fd.Deltaiw, 1j*self.fd.dlr_if,
                                              eps=epstol, Np_max=Npmax, Hermitian=Hermitian)

            if is_root() and verbose:
                diff = np.max(np.abs(delta_iaa + g_iaa_reconstruct(pol*self.beta, weights, self.tau_i/self.beta)))
                print(f"Time domain diff {diff:2.2E}")
        
            if error < epstol and len(pol)<len(self.tau_i):
                if is_root() and verbose:
                    print("using aaa poles, number of poles is ",len(pol))
                self.fd.copy_aaa_result(pol, weights,-pol,weights)
                self.fd.hyb_decomposition(poledlrflag,eps = fittingeps/10)
            else:
                if is_root() and verbose:
                    print("using dlr poles")
            
                self.fd.hyb_init(delta_iaa,poledlrflag=True)
                self.fd.hyb_decomposition(poledlrflag=True,eps = fittingeps/10)


    def energyshift_bisection(self, Sigma_t, verbose=True):
        
        def target_function(eta_h):
            G_new_eta=self.fd.time_ordered_dyson(self.beta,self.H_mat,eta_h,Sigma_t)
            Z_h = self.fd.partition_function(G_new_eta)
            Omega_h = np.log(np.abs(Z_h))/self.beta
            return Omega_h
        Omega = target_function(self.eta)
        if is_root() and verbose:
            print("Current Omega is ",Omega)
        if np.abs(Omega)>0:
            E_max = self.eta.real if Omega<0. else 0.5*self.lamb/self.beta
            E_min = self.eta.real if Omega>0. else 0.
            bracket=[E_min,E_max]
            sol=root_scalar(target_function, method='brenth', fprime=False, bracket=bracket, rtol=1e-10,options={'disp': True})
            self.eta = sol.root
            if not sol.converged and is_root():
                print("Energy shift failed")
                print(sol)

                
    def solve(self, max_order, tol=1e-9, maxiter=10, update_eta_exact=True , mix=1.0, verbose=True):

        for iter in range(maxiter):
            
            #calculate pseudo-particle self energy diagrams
            Sigma_t = Sigma_calc_loop(self.fd, self.G_iaa, max_order, verbose=True)

            if update_eta_exact:
                #decide eta through bisection
                self.energyshift_bisection(Sigma_t, verbose=verbose)
            
                #solver pseudo-particle Dyson equatipon
                G_iaa_new = self.fd.time_ordered_dyson(self.beta,self.H_mat,self.eta,Sigma_t)
            else:
                G_iaa_new = self.fd.time_ordered_dyson(self.beta, self.H_mat, self.eta, Sigma_t)
                Z = self.fd.partition_function(G_iaa_new)
                deta = np.log(Z) / self.beta
                G_iaa_new[:] *= np.exp(-self.tau_i * deta)[:, None, None]
                self.eta += deta

            if is_root():
                # Expect Z = 1
                Z = self.fd.partition_function(G_iaa_new)
                print(f"Z = {Z}")
            
            diff = np.max(np.abs(self.G_iaa - G_iaa_new))
            
            self.G_iaa = mix*G_iaa_new + (1-mix)*self.G_iaa 
            self.Sigma_iaa = Sigma_t

            if is_root(): print(f'PPSC: iter = {iter:3d} diff = {diff:2.2E}')
            if diff < tol: break


    def calc_spgf(self, max_order, verbose=True):
        
        n_g = self.F.shape[0]
        g_iaa = G_calc_loop(self.fd, self.G_iaa, max_order, n_g, verbose=verbose)
        
        return g_iaa
        
