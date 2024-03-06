import numpy as np

from mpi4py import MPI as mpi

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization
from scipy.optimize import root_scalar
import time
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
def Sigma_calc_loop(fd, G_iaa,max_order,verbose=True):
    assert( max_order >= 1 )
    assert( type(fd) == Fastdiagram )
    if verbose:
        start_time = time.time()    

    Sigma_t = np.zeros((G_iaa.shape[0], G_iaa.shape[1], G_iaa.shape[2]),dtype=np.complex128)
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
            Sigma_t -= sign * fd.Sigma_calc_group(G_iaa, diag_vec, diag_idx_vec)
    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, Sigma_t)
    if is_root() and verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time spent is ",elapsed_time)
    return Sigma_t

def G_calc_loop(fd, G_iaa, max_order,n_g,verbose=True):
    if verbose:
        start_time = time.time() 
    g_iaa = np.zeros((G_iaa.shape[0], n_g, n_g),dtype=np.complex128)
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
            g_iaa -= sign * fd.G_calc_group(G_iaa, diag_vec, diag_idx_vec)
    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, g_iaa) 

    if is_root() and verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time spent is ",elapsed_time)
    return g_iaa

from .ac_pes import polefitting

def g_iaa_reconstruct(poles,weights,tau_i):
    G = np.zeros((tau_i.shape[0],weights.shape[1],weights.shape[1]), dtype = np.complex128)
    
    from pydlr import kernel
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
    
    def set_hybridization(self,poledlrflag,delta_iaa,delta_diff = 1.0,fittingeps = 2e-6,printing=True):
        if poledlrflag == True:
           
            self.fd.hyb_init(delta_iaa,poledlrflag=True)
            self.fd.hyb_decomposition(poledlrflag=True)
            
        else:
            #decomposition and reflection of Delta(t) using aaa poles
            
            self.fd.hyb_init(delta_iaa,poledlrflag)
            epstol=min(fittingeps,delta_diff/100)
            Npmax = len(self.fd.dlr_if)-1
            weights, pol, error = polefitting(self.fd.Deltaiw, 1j*self.fd.dlr_if,eps= epstol,Np_max = Npmax,Hermitian=False)
            if is_root() and printing:
                print("Error in time domain")
                
                print(np.max(np.abs(delta_iaa + g_iaa_reconstruct(pol*self.beta,weights,self.tau_i/self.beta))))
            
            
        
            if error<epstol and len(pol)<len(self.tau_i):
                if is_root() and printing:
                    print("using aaa poles, number of poles is ",len(pol))
                self.fd.copy_aaa_result(pol, weights,-pol,weights)
                self.fd.hyb_decomposition(poledlrflag)
            else:
                if is_root() and printing:
                    print("using dlr poles")
            
                self.fd.hyb_init(delta_iaa,poledlrflag=True)
                self.fd.hyb_decomposition(poledlrflag=True)


  
    def energyshift_bisection(self,Sigma_t,verbose=True):
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

    def solve(self, max_order, ppsc_tol=1e-9, ppsc_maxiter=10, update_eta_exact = True , mix=1.0, verbose=True):
        for ppsc_iter in range(ppsc_maxiter):
            
            #calculate pseudo-particle self energy diagrams
            Sigma_t = Sigma_calc_loop(self.fd, self.G_iaa,max_order,verbose=True)

            if update_eta_exact:
                #decide eta through bisection
                self.energyshift_bisection(Sigma_t,verbose=True)
            
                #solver pseudo-particle Dyson equatipon
                G_iaa_new = self.fd.time_ordered_dyson(self.beta,self.H_mat,self.eta,Sigma_t)
            else:
                G_iaa_new = self.fd.time_ordered_dyson(self.beta, self.H_mat, self.eta, Sigma_t)
                Z = self.fd.partition_function(G_iaa_new)
                deta = np.log(Z) / self.beta
                G_iaa_new[:] *= np.exp(-self.tau_i * deta)[:, None, None]
                self.eta += deta

            #check what we get is partition function ==1
            if is_root(): print(self.fd.partition_function(G_iaa_new))
            
            ppsc_diff = np.max(np.abs(self.G_iaa - G_iaa_new))

            
            self.G_iaa = mix*G_iaa_new + (1-mix)*self.G_iaa 

            if is_root(): print(f'PPSC: iter = {ppsc_iter:3d} diff = {ppsc_diff:2.2E}')
            if ppsc_diff < ppsc_tol: break
            


    def calc_spgf(self, max_order, verbose=True):
        
        n_g = self.F.shape[0]
        g_iaa = G_calc_loop(self.fd, self.G_iaa, max_order, n_g, verbose=verbose)
        
        return g_iaa
        
