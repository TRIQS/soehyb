################################################################################
#
# triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2025 by H. U.R. Strand
#
# triqs_soehyb is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_soehyb is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_soehyb. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################


import numpy as np


from triqs.gf import MeshDLR, MeshDLRImTime, BlockGf
from triqs.operators import c, Operator


from triqs_soehyb.solver import Solver, is_root


class TriqsSolver:

    r""" TRIQS Sum-Of-Exponentials bold HYBridization expansion impurity solver (triqs_soehyb)

    Parameters
    ----------

    beta : double
        inverse temperature

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the Green's functions. It must be a
        list of pairs, each containing the name of the
        Green's function block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    eps : double
        Accuracy of the Discrete Lehmann Representation (DLR) imaginary time basis

    w_max : double
        Energy cut-off of the of the Discrete Lehmann Representation (DLR) imaginary time basis

    verbose : bool, optional
        Verbose printouts (default: `True`)

    """

    def __init__(self, beta, gf_struct, eps, w_max, verbose=True):

        self.verbose = verbose
        
        self.beta = beta
        self.gf_struct = gf_struct
        self.eps = eps
        self.w_max = w_max
        
        self.dmesh = MeshDLR(beta=beta, statistic='Fermion', eps=eps, w_max=w_max)        
        self.tmesh = MeshDLRImTime(beta=beta, statistic='Fermion', eps=eps, w_max=w_max)        

        self.Delta_tau = BlockGf(mesh=self.tmesh, gf_struct=self.gf_struct)
        
        # gf_struct -> fundamental_operators
        
        fundamental_operators = []
        for s, n in gf_struct:
            fundamental_operators += [ c(s, i) for i in range(n) ]
            
        H_loc = 0 * Operator()
        
        lamb = beta * w_max
        self.S = Solver(beta, lamb, eps, H_loc, fundamental_operators, verbose=verbose)
        
        np.testing.assert_array_almost_equal(self.dmesh.values(), self.S.dlr_rf)
        np.testing.assert_array_almost_equal(self.tmesh.values(), self.S.tau_i)


    def solve(self, h_int, order, **kwargs):

        r""" Self-consistent solution of the pseudo-particle Green's function
        and pseudo-particle self-energy.

        Parameters
        ----------

        h_int : Triqs Operator
            Local many-body Hamiltonian of the impurity problem

        order : int
            Expansion order of the bold hybridization expansion

        tol : float, optional
            Pseudo-particle self-consistency convergence tolerance (default: `1e-9`)

        maxiter : int, optional
            Maximal number of self-consistent iterations (default: `10`)

        update_eta_exact : bool, optional
            Pseudo-particle energy shift update strategy (default: `True`)

        mix : float, optional
            Linear mixing ratio in the range [0, 1] (default: `1.0`)

        verbose : bool, optional
            Verbose printouts (default: `True`)

        G0_iaa : ndarray/None, optional
            Initial guess for the pseudo-particle propagator (default: `None`)

        """
        
        self.order = order
        self.h_int = h_int
        
        self.S.set_H_loc(h_int)
        self.S.G_iaa = self.S.G0_iaa.copy() # Fixme: use S.__setup_initial_guess?

        self.delta_iaa = self.__from_blockgf_to_array(self.Delta_tau)
        self.S.set_hybridization(self.delta_iaa, compress=True)

        if is_root():
            order_n_diags = [ (o, self.S.fd.number_of_diagrams(o)) for o in range(1,order+1) ]
            print(f'(Order, N_Diags) = {order_n_diags}')
            print(f'max_order = {order}')
        
        self.S.solve(order, **kwargs)

        self.g_iaa = self.S.calc_spgf(order, verbose=False)
        self.G_tau = self.__from_array_to_blockgf(self.g_iaa)

        if is_root():
            print(); self.S.timer.write()      

                
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
        argv_keys = ['verbose']
        verbose = d['verbose']
        d['verbose'] = False # -- Suppress printouts on reconstruction from dict
        ret = cls(*[ d[key] for key in arg_keys ],
                  **{ key : d[key] for key in argv_keys })
        ret.__dict__.update(d)
        ret.verbose = verbose
        return ret


# -- Register Solver in Triqs formats

from h5.formats import register_class
register_class(TriqsSolver)
