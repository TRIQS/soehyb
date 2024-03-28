
import time

import numpy as np
from scipy.optimize import root_scalar

from mpi4py import MPI as mpi

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from .pycppdlr import build_dlr_rf
from .pycppdlr import ImTimeOps

from .impurity import Fastdiagram, DysonItPPSC
from .ac_pes import polefitting, kernel
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


def Sigma_calc_loop(fd, G_iaa, max_order, verbose=True):

    assert( max_order >= 1 )
    assert( type(fd) == Fastdiagram )
    
    if verbose:
        start_time = time.time()    

    Sigma_t = np.zeros((G_iaa.shape[0], G_iaa.shape[1], G_iaa.shape[2]), dtype=complex)
    
    for ord in range(1, max_order+1):
        n_diags = fd.number_of_diagrams(ord)
        
        if is_root() and verbose:
            print(f"PPSC: Sigma order = {ord}, n_diags = {n_diags}")

        for sign, diag in all_connected_pairings(ord):

            #if is_root() and verbose: print(sign, diag)
            
            diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
            diag_idx_vec = np.arange(n_diags, dtype=np.int32)
            diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
            Sigma_t += pow(-1,ord)* sign * fd.Sigma_calc_group(G_iaa, diag_vec, diag_idx_vec)

    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, Sigma_t)

    if is_root() and verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"PPSC: Sigma time {elapsed_time:2.2E}s.")

    return Sigma_t


def G_calc_loop(fd, G_iaa, max_order, n_g, verbose=True):

    if verbose:
        start_time = time.time()
    
    g_iaa = np.zeros((G_iaa.shape[0], n_g, n_g), dtype=complex)

    for ord in range(1, max_order+1):
        n_diags = fd.number_of_diagrams(ord)

        if is_root() and verbose:
            print(f"PPSC: SPGF order = {ord}, n_diags = {n_diags}")
            
        for sign, diag in all_connected_pairings(ord):

            #if is_root(): print(sign, diag)
                
            diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
            diag_idx_vec = np.arange(n_diags, dtype=np.int32)
            diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
            g_iaa += pow(-1,ord)* sign * fd.G_calc_group(G_iaa, diag_vec, diag_idx_vec)

    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, g_iaa) 

    if is_root() and verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"PPSC: SPGF time {elapsed_time:2.2E}s.")
        
    return g_iaa


def eval_dlr_freq(G_xaa, z,  beta, dlr_rf):
    w_x = dlr_rf / beta
    kernel_zx = 1./(z[:, None] - w_x[None, :])
    G_zaa = np.einsum('zx,xab->zab',kernel_zx, G_xaa)
    return G_zaa

class Solver(object):

    def __init__(self, beta, lamb, eps, H_loc, fundamental_operators,ntau=100):

        self.lamb = lamb
        self.eps = eps
        
        self.dlr_rf = build_dlr_rf(lamb, eps)
        self.ito = ImTimeOps(lamb, self.dlr_rf)
    
        self.beta = beta
        self.H_loc = H_loc
        self.fundamental_operators = fundamental_operators

        self.ed = TriqsExactDiagonalization(H_loc, fundamental_operators, beta)

        self.F_dag = np.array([
            np.array(self.ed.rep.sparse_operators.c_dag[idx].todense())
            for idx in range(len(fundamental_operators))])

        self.F = np.array([np.array(
            self.ed.rep.sparse_operators.c_dag[idx].T.conj().todense())
            for idx in range(len(fundamental_operators)) ])
        
        self.H_mat = np.array(self.ed.ed.H.todense())

        self.fd = Fastdiagram(beta, lamb, eps, self.F, self.F_dag)
        self.tau_i = self.fd.get_it_actual().real
        
        # if type(G0_iaa) == np.ndarray and len(G0_iaa.shape) == 3:
        #     if is_root(): print("PPSC: Starting from given G0_iaa")
        #     self.G0_iaa = G0_iaa
        # else:
        #     self.G0_iaa = self.fd.free_greens_ppsc(beta, self.H_mat)
        self.G0_iaa = self.fd.free_greens_ppsc(beta, self.H_mat)
        self.G_iaa = self.G0_iaa.copy()
        self.eta = 0.

        self.dyson = DysonItPPSC(beta, self.ito, self.H_mat)

       

        self.tau_f = np.linspace(0, self.beta, num=ntau)
        def interp(g_xaa):
            eval = lambda t : self.ito.coefs2eval(g_xaa, t/self.beta)
            return np.vectorize(eval, signature='()->(m,m)')(self.tau_f)
        self.interp = interp

        
    def set_hybridization(self, delta_iaa,
                          compress=False, delta_diff=1.0, fittingeps=2e-6,
                          Hermitian=False, verbose=True):
        
        """Set the hybridization function of the expansion.

        Parameters
        ----------

        delta_iaa : (n, m, m) ndarray
            Hybridization function on imaginary time DLR nodes.

        compress : bool, optional
            The hybridization is by default represented using the DLR poles. When enabling
            compression the representation is reduced to an (if possible) even smaller
            number of customized poles, using the AAA algorithm.
            Default `False`

        delta_diff : float, optional
            Current maximal difference between DMFT self-consistent steps in the  hybridization function. 
            The pole fitting tolerance `fittingeps` will be constrained to `fittingeps < delta_diff / 1000`
            to ensure the pole fitting error will not affect the self-consistent iterations.
            In one-shot impurity solvers, one can neglect this argument. 
            Default 1.0

        fittingeps : float, optional
            The pole fitting error tolerance
            Default 2e-6

        Hermitian : bool, optional
            Enforce the hybridization representation to be Hermitian. 
            Default `False`

        verbose : bool, optional
            Enable (more) verbose printouts of the method.
            Defailt `True`
        
        """
        
        if compress == False:        
            self.fd.hyb_init(delta_iaa, poledlrflag=True)
            self.fd.hyb_decomposition(poledlrflag=True, eps=fittingeps/10)
            
        else:
            # decomposition and reflection of Delta(t) using aaa poles
            delta_xaa = self.ito.vals2coefs(delta_iaa) 
            self.fd.hyb_init(delta_iaa, poledlrflag=False)
            epstol = min(fittingeps, delta_diff/100)

            dlr_if_dense = self.fd.dlr_if_dense

            Deltat = self.interp(delta_xaa)
            Deltaiw_dense = eval_dlr_freq(delta_xaa, 1j*dlr_if_dense, self.beta, self.dlr_rf)
            Npmax = Deltaiw_dense.shape[0] -1
            weights, pol, error = polefitting(
                Deltaiw_dense, 1.j*dlr_if_dense, delta_iaa, self.tau_i, Deltat, self.tau_f,self.beta,
                eps=epstol, Np_max=Npmax, Hermitian=Hermitian)

            if is_root() and verbose:
                
                print(f"PPSC: Hybridization fit tau-diff {error:2.2E}")
        
            if error < epstol and len(pol)<=len(self.tau_i):
                if is_root() and verbose:
                    print(f"PPSC: Hybridization using {len(pol)} AAA poles.")
                
                weights_reflect = weights.copy()
                for i in range(weights.shape[0]):
                    weights_reflect[i,:,:] = np.transpose(weights[i,:,:])
                self.fd.copy_aaa_result(pol, weights)
                self.fd.hyb_decomposition(poledlrflag=False, eps=fittingeps/10)
            else:
                if is_root() and verbose:
                    print("PPSC: Hybridization using all DLR poles.")
            
                self.fd.hyb_init(delta_iaa, poledlrflag=True)
                self.fd.hyb_decomposition(poledlrflag=True, eps=fittingeps/10)


    def energyshift_bisection(self, Sigma_iaa, verbose=True):
        
        def target_function(eta_h):
            G_iaa_new = self.dyson.solve(Sigma_iaa, eta)
            Z_h = self.fd.partition_function(G_iaa_new)
            Omega_h = np.log(np.abs(Z_h)) / self.beta            
            return Omega_h
        
        Omega = target_function(self.eta)

        if is_root() and verbose:
            print(f'PPSC: Eta bisection: Z-1 = {Z-1:+2.2E}, Omega = {Omega:+2.2E}')

        if np.abs(Omega) > 0:
            
            E_max = self.eta.real if Omega < 0. else 0.5*self.lamb/self.beta
            E_min = self.eta.real if Omega > 0. else 0.
            
            bracket=[E_min, E_max]
            
            sol = root_scalar(target_function, method='brenth',
                              fprime=False, bracket=bracket, rtol=1e-10, options={'disp': True})
            
            if not sol.converged and is_root():
                print("PPSC: Warning! Energy shift failed.")
                print(sol)

            return sol.root


    def energyshift_newton(self, Sigma_iaa, tol=1e-10, verbose=True):

        def target_function(eta):
        
            G_iaa_new = self.dyson.solve(Sigma_iaa, eta)
            
            Z = self.fd.partition_function(G_iaa_new)
            Omega = np.log(np.abs(Z)) / self.beta

            if verbose and is_root():
                print(f'PPSC: Eta Newton: Z-1 = {Z-1:+2.2E}, Omega = {Omega:+2.2E}')

            G_xaa = self.ito.vals2coefs(G_iaa_new)
            GG_xaa = self.ito.convolve(self.beta, "cppdlr::Fermion", G_xaa, G_xaa, True)
            TrGGb = self.fd.partition_function(GG_xaa)
            dOmega = TrGGb / self.beta / Z

            return Omega, dOmega

        sol = root_scalar(
            target_function, x0=self.eta, method='newton', fprime=True, rtol=tol)

        if not sol.converged and is_root():
            print('PPSC: Warning! Energy shift Newton search failed.')
            print(sol)

        if not sol.converged:
            return self.energyshift_bisection(Sigma_iaa, verbose=verbose)
        
        return sol.root
        

    def solve(self, max_order, tol=1e-9, maxiter=10, update_eta_exact=True, mix=1.0, verbose=True, G0_iaa = None):
        if type(G0_iaa) == np.ndarray and len(G0_iaa.shape) == 3:
            self.G_iaa = G0_iaa
        for iter in range(maxiter):
            
            Sigma_iaa = Sigma_calc_loop(self.fd, self.G_iaa, max_order, verbose=verbose)

            if verbose:
                dyson_start_time = time.time()
                
            if update_eta_exact:
                self.eta = self.energyshift_newton(Sigma_iaa, tol=0.1*tol, verbose=verbose)
                G_iaa_new = self.dyson.solve(Sigma_iaa, self.eta)
                
            else:
                G_iaa_new = self.dyson.solve(Sigma_iaa, self.eta)
                
                Z = self.fd.partition_function(G_iaa_new)
                deta = np.log(np.abs(Z)) / self.beta
                G_iaa_new[:] *= np.exp(-self.tau_i * deta)[:, None, None]
                self.eta += deta
                
            if is_root() and verbose:
                dyson_end_time = time.time()
                dyson_elapsed_time = dyson_end_time - dyson_start_time
                print(f"PPSC: Dyson time {dyson_elapsed_time:2.2E}s.")
                
            if is_root():
                # Expect Z = 1
                Z = self.fd.partition_function(G_iaa_new)
                print(f"PPSC: Z-1 = {Z-1:+2.2E}")

            diff = np.max(np.abs(self.G_iaa - G_iaa_new))
            
            self.G_iaa = mix*G_iaa_new + (1-mix)*self.G_iaa 
            self.Sigma_iaa = Sigma_iaa

            if is_root(): print(f'PPSC: iter = {iter:3d} diff = {diff:2.2E}')
            if diff < tol: break


    def calc_spgf(self, max_order, verbose=True):
        
        n_g = self.F.shape[0]
        g_iaa = G_calc_loop(self.fd, self.G_iaa, max_order, n_g, verbose=verbose)
        
        return g_iaa
        
