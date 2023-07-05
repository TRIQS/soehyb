#pragma once
#include "nda/nda.hpp"
#include "cppdlr/cppdlr.hpp"

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