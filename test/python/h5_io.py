""" Author: Hugo U. R. Strand (2024) """

from triqs_soehyb.solver import Solver

from triqs.operators import c, c_dag

from h5 import HDFArchive


def test_h5():

    beta = 1.0
    lamb = 100.
    eps = 1e-12
    
    H = c_dag(0,0) * c(0,0)
    fundamental_operators = [ c(0,i) for i in range(1) ]

    S = Solver(beta, lamb, eps, H, fundamental_operators)
        
    filename = 'data_h5_io.h5'
    
    with HDFArchive(filename, 'w') as A: A['S'] = S
    with HDFArchive(filename, 'r') as A: S_ref = A['S']
        
    assert( S == S_ref )
        
    
if __name__ == '__main__':

    test_h5()
