#pragma once
#include "nda/nda.hpp"
#include "cppdlr/cppdlr.hpp"
#include <nda/blas/tools.hpp>
#include "strong_cpl.hpp"

using namespace cppdlr;
using namespace nda;

/**
  * @class fastdiagram
  * @brief Class responsible for fast diagram calculation of a given impurity problem using hybridization expansion.
  */
class fastdiagram{
    public:
    /** 
    * @brief Constructor for fastdiagram
    */
    fastdiagram(double beta, double lambda, double eps, nda::array<dcomplex,3> Deltat,nda::array<dcomplex,3> F, nda::array<dcomplex,3> F_dag, bool poledlrflag) ;
    nda::array<dcomplex,3> Sigma_calc(nda::array_const_view<dcomplex,3> Gt, std::string order);
    nda::array<dcomplex,3> G_calc(nda::array_const_view<dcomplex,3> Gt, std::string order); 

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

    nda::array<int,2> D_NCA;
    nda::array<int,2> D_OCA;
    nda::array<int,2> D_TCA_1;
    nda::array<int,2> D_TCA_2;
    nda::array<int,2> D_TCA_3;
    nda::array<int,2> D_TCA_4;
};


