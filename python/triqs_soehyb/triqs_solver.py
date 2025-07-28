



import numpy as np


from triqs.gf import MeshDLR, MeshDLRImTime, BlockGf
from triqs.operators import c, Operator

from triqs_soehyb.solver import Solver


class TriqsSolver:

    def __init__(self, beta, gf_struct, eps, w_max):

        self.beta = beta
        self.gf_struct = gf_struct
        self.eps = eps
        self.w_max = w_max
        
        print("--> triqs_soehyb.triqs_solver.TriqsSolver")

        self.dmesh = MeshDLR(beta=beta, statistic='Fermion', eps=eps, w_max=w_max)        
        self.tmesh = MeshDLRImTime(beta=beta, statistic='Fermion', eps=eps, w_max=w_max)        

        self.Delta_tau = BlockGf(mesh=self.tmesh, gf_struct=self.gf_struct)
        
        # gf_struct -> fundamental_operators

        print(f'gf_struct = {gf_struct}')
        
        fundamental_operators = []
        for s, n in gf_struct:
            fundamental_operators += [ c(s, i) for i in range(n) ]
            
        print(f'fundamental_operators = {fundamental_operators}')
        
        H_loc = 0 * Operator()

        print(f'H_loc = {H_loc}')
        
        lamb = beta * w_max
        self.S = Solver(beta, lamb, eps, H_loc, fundamental_operators)

        np.testing.assert_array_almost_equal(self.dmesh.values(), self.S.dlr_rf)
        np.testing.assert_array_almost_equal(self.tmesh.values(), self.S.tau_i)


    def solve(self, h_int, order, **kwargs):

        self.order = order
        self.h_int = h_int
        
        self.S.set_H_loc(h_int)
        self.S.G_iaa = self.S.G0_iaa.copy() # Fixme: use S.__setup_initial_guess?

        self.delta_iaa = self.__from_blockgf_to_array(self.Delta_tau)
        self.S.set_hybridization(self.delta_iaa, compress=True)

        self.S.solve(order, **kwargs)

        self.g_iaa = self.S.calc_spgf(order)        
        self.G_tau = self.__from_array_to_blockgf(self.g_iaa)

                
    def __from_blockgf_to_array(self, G):

        for b, g in G:
            assert( len(g.target_shape) == 2)
            assert( g.target_shape[0] == g.target_shape[1] )
            
        norb = sum([ g.target_shape[0] for b, g in G ])

        assert( norb == len(self.S.fundamental_operators) )
        
        ntau = len(G.mesh)
        g_iaa = np.zeros((ntau, norb, norb), dtype=complex)
        
        sidx = 0
        for b, g in G:
            size = g.target_shape[0]
            g_iaa[:, sidx:sidx+size, sidx:sidx+size] = g.data
            sidx += size

        return g_iaa

        
    def __from_array_to_blockgf(self, g_iaa):

        G = BlockGf(mesh=self.tmesh, gf_struct=self.gf_struct)
    
        sidx = 0
        for b, g in G:
            size = g.target_shape[0]
            g.data[:] = g_iaa[:, sidx:sidx+size, sidx:sidx+size]
            sidx += size

        return G


    def __skip_keys(self):
        return []


    def __reduce_to_dict__(self):
        d = self.__dict__.copy()
        keys = set(d.keys()).intersection(self.__skip_keys())
        for key in keys: del d[key]
        return d


    @classmethod
    def __factory_from_dict__(cls, name, d):
        arg_keys = ['beta', 'gf_struct', 'eps', 'w_max']
        argv_keys = []
        ret = cls(*[ d[key] for key in arg_keys ],
                  **{ key : d[key] for key in argv_keys })
        ret.__dict__.update(d)
        return ret


# -- Register Solver in Triqs formats

from h5.formats import register_class
register_class(TriqsSolver)
