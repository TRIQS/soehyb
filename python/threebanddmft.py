
from itertools import product
import time
import numpy as np
from triqs_soehyb.diag import *
from triqs_soehyb.ac_pes import *
from h5 import HDFArchive
from mpi4py import MPI as mpi

from triqs.operators import c, c_dag, n, Operator
from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization


from scipy.integrate import quad

from triqs_soehyb.pycppdlr import build_dlr_rf
from triqs_soehyb.pycppdlr import ImTimeOps

from triqs_soehyb.impurity import Fastdiagram
from scipy.optimize import root_scalar
def kernel(tau, omega):
    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

    return kernel
def H_soc_from_levi_cevita(lamb_soc):

    """ Constructs the Triqs operator for spin-orbit coupling
    in the cubic harmonic t2g basis.

    H_{soc} = \sum_{abc} \sum_{s_1, s_2}
        \epsilon_{abc} \sigma^{c}_{s_1, s_2} c^\dagger_{a s_1} c_{b s_2}

    where $a,b,c \in \{x, y, z\}$ and/or $\{yz, xz, xy\}$
    
    The orbital indices [0, 1, 2] are mapped to the cubic harmonics
    according to [0, 1, 2] -> [yz, xz, xy].

    Author: Hugo U. R. Strand """

    norb = 3
    spin_names = ('up','do')
    orb_names = list(range(norb))
    
    fops = [(sn,on) for sn, on in product(spin_names, orb_names)]
    fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]
    #print(f'fundamental_operators = \n{fundamental_operators}')

    sigma_x = np.array([[0., 1.], [1., 0.]])
    sigma_y = np.array([[0., -1.j], [1.j, 0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])
    sigma_vec = np.array([sigma_x, sigma_y, sigma_z])
    
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    H_soc = Operator()
    for i, j, k in product(range(norb), repeat=3):
        for s1, s2 in product(range(2), repeat=2):
            S1, S2 = spin_names[s1], spin_names[s2]
            H_soc += lamb_soc * 0.5j * \
                eijk[i, j, k] * sigma_vec[k, s1, s2] * c_dag(S1, i) * c(S2, j)

    return H_soc

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

def eval_semi_circ_tau(tau, beta, h, t):
    I = lambda x : -2 / np.pi / t**2 * \
        kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]
    g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
    return g

eval_semi_circ_tau = np.vectorize(eval_semi_circ_tau)

def g_iaa_reconstruct(poles,weights,tau_i):
    G = np.zeros((tau_i.shape[0],weights.shape[1],weights.shape[1]), dtype = np.complex128)
    
    from pydlr import kernel
    for n in range(poles.shape[0]): 
        for i in range(tau_i.shape[0]):
            # print(i)
            G[i,:,:] = G[i,:,:] + kernel(tau_i[i:i+1],poles[n:n+1])[0,0]*weights[n,:,:]
    return G


def run_calc(beta, lamb, eps, order,
             mu=3.9530058540332917,
             lamb_soc=0.,
             write_h5=False, plot_flag=True):
    max_order=order
    U = 4.6
    J = 0.8
    norb = 3
    
    half_filling_shift = 0.5*(5*U - 10*J)
    mu += half_filling_shift
    t_diag = np.array([0.5, 0.5, 1.0])
    T_diag = np.concatenate([t_diag, t_diag])
    delta_cf = -1.0
    
    ppsc_tol = 1e-6
    ppsc_maxiter = 1
    dmft_tol = 1e-6
    dmft_maxiter = 100 # self-cons settings
    use_symmetry = False

    order_dict = {
        1 : ('nca', False, False),
        2 : ('oca', True, False),
        3 : ('tca', True, True),
        }

    order, oca, tca = order_dict[order]
    
    verbose = False

    spin_names = ('up','do')
    orb_names = list(range(norb))
    
    fops = [(sn,on) for sn, on in product(spin_names, orb_names)]
    fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]

    KanMat1, KanMat2 = U_matrix_kanamori(norb, U, J)
    H = h_int_kanamori(spin_names, norb, KanMat1, KanMat2, J, True)

    N_0 = n('up', 0) + n('do', 0)
    N_2 = n('up', 2) + n('do', 2)
    N_up = sum([ n('up', idx) for idx in range(norb) ])
    N_do = sum([ n('do', idx) for idx in range(norb) ])

    H -= mu *(N_up + N_do)
    H += delta_cf * N_2 # using that orbital 2 is xy

    H += H_soc_from_levi_cevita(lamb_soc)
    
    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)
    
    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)
    r = ito.rank()
    if is_root(): print(f'N_dlr = {r}')

    F = np.array([np.array(
        ed.rep.sparse_operators.c_dag[idx].T.conj().todense())
        for idx in range(len(fundamental_operators)) ])

    F_dag = np.array([np.array(
        ed.rep.sparse_operators.c_dag[idx].todense())
        for idx in range(len(fundamental_operators)) ])
    
    fd0 = Fastdiagram(beta, lamb, eps, F, F_dag)
    
    tau_i = fd0.get_it_actual().real

    delta_iaa = np.zeros((r, 2*norb, 2*norb), dtype=ed.ed.H.dtype)
    for idx, t_i in enumerate(T_diag):
        delta_iaa[:, idx, idx] = \
            t_i**2 * eval_semi_circ_tau(tau_i, beta, h=0.0, t=t_i)
    
    # Setup impurity problem    
    if is_root(): print("-> Atomic propagator G0")
    H_mat = np.array(ed.ed.H.todense())
    #np.testing.assert_array_almost_equal(H_mat, H_mat.T.conj())
    G0_iaa = fd0.free_greens_ppsc(beta, H_mat)
    #np.testing.assert_array_almost_equal(G0_iaa, np.transpose(G0_iaa.conj(), axes=(0, 2, 1)))
    G_iaa = G0_iaa.copy()    

    g_iaa = np.zeros_like(delta_iaa)
    eta = 0.
    # IW = dlr(lamb=100000,eps=1e-16).get_matsubara_frequencies(beta)
    IW = 1.0j*np.arange(-99,99,2)*np.pi/beta
    delta_diff = np.inf 
    for dmft_iter in range(dmft_maxiter):
        
        fd1 = Fastdiagram(beta, lamb, eps, F, F_dag)
        poledlrflag=True
        fd1.hyb_init(delta_iaa,poledlrflag)
        fd1.hyb_decomposition(poledlrflag)
         #decomposition and reflection of Delta(t) using aaa poles
        poledlrflag = False
        fd2 = Fastdiagram(beta, lamb, eps, F, F_dag)
        fd2.hyb_init(delta_iaa,poledlrflag)
        
        # from pydlr import dlr
        # d1 = dlr(lamb=lamb,eps=eps)
        # deltag = d1.lstsq_dlr_from_tau(tau_i, delta_iaa, beta)
        
       
        # delta_iaa_reflect = d1.eval_dlr_tau(deltag,beta-tau_i,beta)
        # deltag_reflect = d1.lstsq_dlr_from_tau(tau_i, delta_iaa_reflect, beta)
        # breakpoint()
        #*
        # deltaiw_long = d1.eval_dlr_freq(deltag,IW,beta)
        # _, _, residue = get_weight(dlr_rf/beta, IW, deltaiw_long,cleanflag=True)
        # epstol = max(np.linalg.norm(residue)+1e-6/2,1e-6)
        # deltaiw_reflect_long = d1.eval_dlr_freq(deltag_reflect,IW,beta)
        # _, _, residue_r = get_weight(dlr_rf/beta, IW, deltaiw_reflect_long,cleanflag=True)
        # epstol_r = max(np.linalg.norm(residue_r)+1e-6/2,1e-6)
        epstol=min(2e-6,delta_diff/100)
        # epstol_r = epstol*1.0
        # weights, pol, error = polefitting(deltaiw_long, IW,eps= epstol,Np_max = int(len(IW)/2),Hermitian=False)
        # weights_reflect, pol_reflect, error_reflect = polefitting(deltaiw_reflect_long, IW,eps= epstol_r,Np_max = int(len(IW)/2),Hermitian=False)
        Npmax = len(fd2.dlr_if)-1
        weights, pol, error = polefitting(fd2.Deltaiw, 1j*fd2.dlr_if,eps= epstol,Np_max = Npmax,Hermitian=False)
        # weights_reflect, pol_reflect, error_reflect = polefitting(fd2.Deltaiw_reflect, 1j*fd2.dlr_if,eps= epstol_r,Np_max = Npmax,Hermitian=False)
        if is_root():
            print("Error in time domain")
            print(np.max(np.abs(delta_iaa + g_iaa_reconstruct(pol*beta,weights,tau_i/beta))))
            # print(np.max(np.abs(delta_iaa_reflect + g_iaa_reconstruct(-pol*beta,weights,tau_i/beta))))
        # breakpoint()

        # fd2.copy_aaa_result(pol, weights,pol_reflect,weights_reflect)
        fd2.copy_aaa_result(pol, weights,-pol,weights)
        fd2.hyb_decomposition(poledlrflag)
        # if error<epstol and error_reflect<epstol and len(pol)<r and len(pol_reflect)<r:
        if error<epstol and len(pol)<r:
            fd = fd2
            if is_root():
                print("number of poles is ",len(pol))
        else:
            fd = fd1

        # if is_root() and fallback==False:
        #     print("Number of poles is ",len(pol))

        for ppsc_iter in range(ppsc_maxiter):
            start_time = time.time()

            #calculate pseudo-particle self energy diagrams
            Sigma_t = np.zeros((r, G_iaa.shape[1], G_iaa.shape[2]),dtype=np.complex128)
            for ord in range(1, max_order+1):
                n_diags = fd.number_of_diagrams(ord)
                # if is_root():
                    # print(f"order = {order}")
                    # print(f'n_diags = {n_diags}')
                for sign, diag in all_connected_pairings(ord):
                    if  is_root():
                        print(sign, diag)
                    diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
                    diag_idx_vec = np.arange(n_diags, dtype=np.int32)
                    diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
                    Sigma_t -= sign * fd.Sigma_calc_group(G_iaa, diag_vec, diag_idx_vec)



            mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, Sigma_t)
            if is_root():
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Time spent is ",elapsed_time)


            #np.testing.assert_array_almost_equal(Sigma_iaa, np.transpose(Sigma_iaa.conj(), axes=(0, 2, 1)))
            #G_iaa_new = fd.time_ordered_dyson(beta, H_mat, eta, Sigma_t) # has to give eta0 here to get correct G0
            #np.testing.assert_array_almost_equal(G_iaa_new, np.transpose(G_iaa_new.conj(), axes=(0, 2, 1)))

            if True:
                def target_function(eta_h):
                    G_new_eta=fd.time_ordered_dyson(beta,H_mat,eta_h,Sigma_t)
                    Z_h = fd.partition_function(G_new_eta)
                    Omega_h = np.log(np.abs(Z_h))/beta
                    return Omega_h
                Omega = target_function(eta)
                if is_root():
                    print("Current Omega is ",Omega)
                if np.abs(Omega)>0:
                    E_max = eta.real if Omega<0. else 0.5*lamb/beta
                    E_min = eta.real if Omega>0. else 0.
                    bracket=[E_min,E_max]
                    sol=root_scalar(target_function, method='brenth', fprime=False, bracket=bracket, rtol=1e-10,options={'disp': True})
                    eta = sol.root
                    if not sol.converged and is_root():
                        print("Energy shift failed")
                        print(sol)
            
            G_iaa_new = fd.time_ordered_dyson(beta,H_mat,eta,Sigma_t)
            if is_root():
                print(fd.partition_function(G_iaa_new))
                
                
            ppsc_diff = np.max(np.abs(G_iaa - G_iaa_new))

            G_iaa = G_iaa_new

            if is_root(): print(f'PPSC: iter = {ppsc_iter:3d} diff = {ppsc_diff:2.2E}')
            if ppsc_diff < ppsc_tol: break
            
        g_iaa_old = g_iaa

        g_iaa = np.zeros((r, delta_iaa.shape[1], delta_iaa.shape[1]),dtype=np.complex128)
        start_time = time.time()
        for ord in range(1, max_order+1):
            n_diags = fd.number_of_diagrams(ord)
            # if is_root():
                # print(f"order = {order}")
                # print(f'n_diags = {n_diags}')
            for sign, diag in all_connected_pairings(ord):
                if  is_root():
                    print(sign, diag)
                diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
                diag_idx_vec = np.arange(n_diags, dtype=np.int32)
                diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
                g_iaa -= sign * fd.G_calc_group(G_iaa, diag_vec, diag_idx_vec)
        mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, g_iaa) 

        if is_root():
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Time spent is ",elapsed_time)

                
        delta_iaa_old = delta_iaa
        delta_iaa = np.einsum('a,iab,b->iab', T_diag, g_iaa, T_diag)
        # delta_iaa_reflect = ito.reflect()

        delta_diff = np.max(np.abs(g_iaa - g_iaa_old))
        
        G_xaa = ito.vals2coefs(G_iaa)
        rho_GG = -ito.coefs2eval(G_xaa, 1.0)

        g_xaa = ito.vals2coefs(g_iaa)        
        rho_aa = -ito.coefs2eval(g_xaa, 1.0)

        N = np.sum(np.diag(rho_aa)).real
                 
        if is_root():
            print(
                f'DMFT: iter {dmft_iter:3d}, ddelta {delta_diff:2.2E}' + \
                ' - ' + f'PPSC: iter {ppsc_iter:3d}, dppgf {ppsc_diff:2.2E}' + \
                f' - N {N:8.8E}')

        if delta_diff < dmft_tol: break

    def interp(g_xaa, tau_j):
        eval = lambda t : ito.coefs2eval(g_xaa, t/beta)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)


    tau_f = np.linspace(0, beta, num=100)

    g_xaa = ito.vals2coefs(g_iaa)
    g_faa = interp(g_xaa, tau_f)

    delta_xaa = ito.vals2coefs(delta_iaa)
    delta_faa = interp(delta_xaa, tau_f)

    G_xaa = ito.vals2coefs(G_iaa)
    G_faa = interp(G_xaa, tau_f)

    Sigma_xaa = ito.vals2coefs(Sigma_t)
    Sigma_faa = interp(Sigma_xaa, tau_f)
    
    class Dummy():
        def __init__(self):
            pass

    d = Dummy()
    d.N = N
    
    if is_root() and write_h5:
        filename = f'data_cro_fastdiag_{order}_beta_{beta}_soc_{lamb_soc}.h5'
        print(f'--> Writing: {filename}')
        with HDFArchive(filename, 'w') as A:

            A['g_iaa'] = g_iaa
            A['delta_iaa'] = delta_iaa
            A['tau_i'] = tau_i
            
            A['g_faa'] = g_faa
            A['delta_faa'] = delta_faa
            A['tau_f'] = tau_f

            A['G_faa'] = G_faa
            A['Sigma_faa'] = Sigma_faa

            A['rho'] = rho_aa
            
            A['beta'] = beta
            A['order'] = order

            A['lamb'] = lamb
            A['eps'] = eps

            A['H'] = H
            
            A['U'] = U
            A['J'] = J
            A['mu'] = mu
            A['t_diag'] = t_diag
            A['delta_cf'] = delta_cf
            A['lamb_soc'] = lamb_soc
            A['N'] = N

    if is_root() and plot_flag:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        subp = [6, 6, 1]

        nspinorb = 6
        rng = [0, 1, 5, 3, 4, 2]
        lables = [
            r'$yz \uparrow$',
            r'$xz \uparrow$',
            r'$xy \uparrow$',
            r'$yz \downarrow$',
            r'$xz \downarrow$',
            r'$xy \downarrow$',
            ]
        for a, b in product(rng, repeat=2):
            plt.subplot(*subp); subp[-1] += 1
            la, lb = lables[a], lables[b]
            plt.title(la + ', ' + lb, fontsize=7)
            for flt in [np.real, np.imag]:
                l = plt.plot(
                    tau_i, flt(g_iaa[:, a, b]),
                    '.', label=f'g {a},{b}', alpha=0.5)
                color = l[0].get_color()
                plt.plot(
                    tau_f, flt(g_faa[:, a, b]),
                    '-', color=color, alpha=0.75)
            #plt.xlabel(r'$\tau$')
            if a != b: plt.ylim([-0.1, 0.1])
            else: plt.ylim([-1, 0.05])

        plt.tight_layout()
        plt.savefig(
            f'figure_fastdiag_cro_order_{order}_lamb_soc_{lamb_soc}.pdf')
        plt.show()

    #return ppsc
    return d

            
def fix_N_calc(N_target, mu0, mu1, func):

    def target_function(mu):
        N = func(mu)
        print('='*72)
        print(f'mu = {mu}, N = {N}')
        print('='*72)
        return N_target - N

    from scipy.optimize import root_scalar

    sol = root_scalar(
        target_function, bracket=[mu0, mu1],
        x0=mu0, x1=mu1, method='bisect', xtol=1e-3)
    
    return sol.root


if __name__ == '__main__':

    order = 2
    
    parms = [
        ( 5., 100., 1e-8),
        # ( 5., 1000., 1e-8),
        ]

    N_target = 4.0

    mu_nca = 3.977224604033292
    mu_oca = 3.965505854033292
    mu_tca = 3.9655058540332924
    mulist = [mu_nca, mu_oca, mu_tca]
    mu = mulist[order-1]
    #mu = mu_nca
    #mu = mu_oca
    # mu = mu_tca

    mu0 = mu - 0.05
    mu1 = mu + 0.05

    lamb_socs = [0.2]
    
    for beta, lamb, eps in parms:

        for lamb_soc in lamb_socs:

            if is_root():
                print('='*72)
                print('='*72)
                print(f'lamb_soc = {lamb_soc}')
                print('='*72)
                print('='*72)
            
            def N_func(mu, write_h5=False):
                p = run_calc(beta, lamb, eps, order,
                             lamb_soc=lamb_soc,
                             mu=mu,
                             write_h5=write_h5, plot_flag = False)
                return p.N

            #mu = fix_N_calc(N_target, mu0, mu1, N_func)

            if is_root():
                print('='*72)
                print('='*72)
                print('--> Final')
                print(f'mu = {mu}')
                print('='*72)
                print('='*72)

            p = N_func(mu, write_h5=True)
            
        

