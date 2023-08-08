#pragma once
#include "nda/nda.hpp"
#include "cppdlr/cppdlr.hpp"
#include <nda/blas/tools.hpp>
#include "strong_cpl.hpp"

using namespace cppdlr;
using namespace nda;

class impuritysolver{
    public:
    impuritysolver(double beta, double lambda, double eps, nda::array<dcomplex,3> Deltat,nda::array<dcomplex,3> F, nda::array<dcomplex,3> F_dag, bool poledlrflag) ;
    nda::array<dcomplex,3> Sigma_calc(nda::array_const_view<dcomplex,3> Gt, nda::array_const_view<int,2> D);
    nda::array<dcomplex,3> G_calc(nda::array_const_view<dcomplex,3> Gt, nda::array_const_view<int,2> D); 

    private:
    double beta;
    double lambda;
    nda::vector<double> dlr_rf;
    imtime_ops itops;
    nda::vector<double> dlr_it;
    
    nda::array<dcomplex,3> F;
    nda::array<dcomplex,3> F_dag;

    nda::array<dcomplex,3> Deltat;
    nda::array<dcomplex,3> Deltat_reflect;

    hyb_F Delta_F;
    hyb_F Delta_F_reflect;
    };