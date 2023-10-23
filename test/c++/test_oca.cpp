#include "strong_cpl.hpp"
#include "impurity.hpp"
#include "nda/nda.hpp"
#include <cppdlr/cppdlr.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <functional>
#include <gtest/gtest.h>
#include <nda/algorithms.hpp>
#include <nda/basic_functions.hpp>
#include <nda/declarations.hpp>
#include <nda/layout/policies.hpp>
#include <chrono>
#include <nda/matrix_functions.hpp>

using namespace cppdlr;
using namespace nda;
TEST(OCA, G){
    double beta = 1.0;
    double lambda = 100.0;
    double eps = 1.0e-13;
    auto dlr_rf = build_dlr_rf(lambda, eps); // Get DLR frequencies
    auto itops = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
    auto const & dlr_it = itops.get_itnodes();
    int r = itops.rank();
    
    nda::vector<double> dlr_it_actual = dlr_it;
    for (int i = 0;i<r;++i) {if (dlr_it(i)<0) dlr_it_actual(i) = dlr_it(i)+1; }

    double D = -2.0;
    auto Deltat = nda::array<dcomplex,3>(r,1,1);
    Deltat(_,0,0) = exp(-D*dlr_it_actual*beta);
    auto Deltadlr = itops.vals2coefs(Deltat);  
    //reflect Deltat
    auto Deltat_reflect  = itops.reflect(Deltat); 
    auto Deltadlr_reflect = itops.vals2coefs(Deltat_reflect);
 
    //decomposition of hybridization


    double g = 3.0;
    auto Gt = nda::array<dcomplex,3>(r,1,1);
    Gt(_,0,0) = exp(-g*dlr_it_actual*beta);
    auto G_xaa = itops.vals2coefs(Gt);

    auto I_aa = nda::eye<dcomplex>(1);
    
    auto D2 = nda::array<int,2>{{0,2},{1,3}};
    auto fb2 =  nda::vector<int>(2); fb2=0;

    auto F = nda::array<dcomplex,3>(1,1,1); F(0,0,0) = 1.0;
    nda::array<double,1> pol(1);
    pol(0) =  D*beta;
    nda::array<double,1> pol_r = -pol;
    nda::array<dcomplex,3> A(1,1,1);
    A(0,_,_) = eye<dcomplex>(1)/k_it(0,D*beta);

    // auto Delta_decomp = hyb_decomp(Deltadlr,dlr_rf);
    // auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect,dlr_rf);
    auto Delta_decomp = hyb_decomp(A,pol);
    auto Delta_decomp_reflect = hyb_decomp(A,pol_r);
    Delta_decomp.check_accuracy(Deltat, dlr_it);
    
    Delta_decomp_reflect.check_accuracy(Deltat_reflect, dlr_it);

    auto Delta_F = hyb_F(Delta_decomp, dlr_rf, dlr_it, F, F);
    auto Delta_F_reflect = hyb_F(Delta_decomp_reflect, dlr_rf, dlr_it, F, F);

    //calculating diagrams
    auto G_OCAdiagram = G_Diagram_calc_sum_all(Delta_F,Delta_F_reflect,D2,Deltat,Deltat_reflect, Gt,itops,beta, F,  F);
    std::cout<<G_OCAdiagram<<std::endl;
    if (abs(D)<1e-8){
        for (int i=0;i<r;++i) std::cout<<" "<<make_regular((exp(-g*beta) * beta*beta*(1-dlr_it_actual(i))*dlr_it_actual(i)));
    }
    else{
        
        for (int i=0;i<r;++i) std::cout<<" "<<make_regular(exp(-(g+D)*beta)*(1-exp(D*(1-dlr_it_actual(i))))*(1-exp(D*dlr_it_actual(i)))/pow(D,2));
        std::cout<<std::endl;
        for (int i=0;i<r;++i) std::cout<<" "<<make_regular(exp(-D*beta)*exp(-(g-D)*beta)*(1-exp(-D*(1-dlr_it_actual(i))))*(1-exp(-D*dlr_it_actual(i)))/pow(D,2));  
    }
} 