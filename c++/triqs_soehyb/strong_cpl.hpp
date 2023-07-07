#pragma once
#include "nda/nda.hpp"
#include "cppdlr/cppdlr.hpp"
#include <nda/blas/tools.hpp>

using namespace cppdlr;
using namespace nda;


/* This is the class for decomposition of a hybridization function. 
    The decomposition is:
        Delta_ab(iv) = sum_R U(a,R)V(b,R)/(iv-w(R))
    Its components are:
        w:    1*P vector: real-valued poles;
        U,V: dim*P array: complex-valued
    We construct this decomposition using dlr. 
    Expecting this step to be replaced by pole fitting procedures in the future. 
*/
class hyb_decomp {
    public:
    nda::vector<double> w;
    nda::matrix<dcomplex> U;
    nda::matrix<dcomplex> V;
    
    hyb_decomp(nda::array_const_view<dcomplex,3> Delta_dlr, nda::vector_const_view<double> dlr_rf, double eps = 1e-14);
};

/* This is the class for constructing U_tilde and V_tilde, based on hyb_decomp and the F matrices.
    Its components are:
        U_tilde, V_tilde, both of size (r,P,dim,dim)
        c, of size (1,P)
*/
class hyb_F {
    public:
    nda::vector<double> c;
    nda::array<dcomplex,4> U_tilde;
    nda::array<dcomplex,4> V_tilde;

    hyb_F(hyb_decomp &hyb_decomp, nda::vector_const_view<double> dlr_rf, nda::vector_const_view<double> dlr_it, double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag);
};