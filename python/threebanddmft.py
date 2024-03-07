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

from triqs_soehyb.solver import *
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



def eval_semi_circ_tau(tau, beta, h, t):
    I = lambda x : -2 / np.pi / t**2 * \
        kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]
    g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
    return g

eval_semi_circ_tau = np.vectorize(eval_semi_circ_tau)



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


    verbose = True

    spin_names = ('up','do')
    orb_names = list(range(norb))
    
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
    
    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)
    r = ito.rank()
    if is_root(): print(f'N_dlr = {r}')   
    
    tau_i = ito.get_itnodes()*beta
    for i1 in range(r):
        if tau_i[i1]<0: tau_i[i1] = tau_i[i1]+beta 
   
    delta_iaa = np.zeros((r, 2*norb, 2*norb), dtype=np.complex128)
    for idx, t_i in enumerate(T_diag):
        delta_iaa[:, idx, idx] = \
            t_i**2 * eval_semi_circ_tau(tau_i, beta, h=0.0, t=t_i)   

    g_iaa = np.zeros_like(delta_iaa)

    delta_diff = np.inf 
    for dmft_iter in range(dmft_maxiter):
        Impurity = Solver(beta, lamb, eps, H, fundamental_operators)
        Impurity.set_hybridization(poledlrflag=False,delta_iaa = delta_iaa,delta_diff = delta_diff,fittingeps = 2e-6,printing=True)
        Impurity.solve(max_order,  ppsc_maxiter=ppsc_maxiter,ppsc_tol = ppsc_tol, update_eta_exact = True , verbose=verbose)          
        g_iaa_old = g_iaa
        #calculate single-particle Green's functions diagrams
        g_iaa = Impurity.calc_spgf(max_order, verbose=verbose)
        # g_iaa = G_calc_loop(fd, G_iaa, max_order,delta_iaa.shape[1],verbose=True)
        #calculate new hybridization function
        delta_iaa = np.einsum('a,iab,b->iab', T_diag, g_iaa, T_diag)

        delta_diff = np.max(np.abs(g_iaa - g_iaa_old))

       
        g_xaa = ito.vals2coefs(g_iaa)
        rho_aa = -ito.coefs2eval(g_xaa, 1.0)
        N = np.sum(np.diag(rho_aa)).real         
        if is_root():  
            print(
                f'DMFT: iter {dmft_iter:3d}, ddelta {delta_diff:2.2E}' + \
                ' - '  + f' - N {N:8.8E}')

        if delta_diff < dmft_tol: break



    N = np.sum(np.diag(rho_aa)).real
    def interp(g_xaa, tau_j):
        eval = lambda t : ito.coefs2eval(g_xaa, t/beta)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)


    tau_f = np.linspace(0, beta, num=100)

    g_xaa = ito.vals2coefs(g_iaa)
    g_faa = interp(g_xaa, tau_f)

    delta_xaa = ito.vals2coefs(delta_iaa)
    delta_faa = interp(delta_xaa, tau_f)

    G_xaa = ito.vals2coefs(Impurity.G_iaa)
    G_faa = interp(G_xaa, tau_f)

    Sigma_xaa = ito.vals2coefs(Impurity.Sigma_iaa)
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
            
        

