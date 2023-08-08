#include "strong_cpl.hpp"
#include "impurity.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <nda/linalg/matmul.hpp>

impuritysolver::impuritysolver(double beta, double lambda, double eps, nda::array<dcomplex,3> Deltat, nda::array<dcomplex,3> F, nda::array<dcomplex,3> F_dag, bool poledlrflag): Deltat(Deltat), F(F), F_dag(F_dag){
    dlr_rf = build_dlr_rf(lambda, eps); // Get DLR frequencies
    itops = imtime_ops(lambda, dlr_rf); 

    dlr_it = itops.get_itnodes();
    Deltat_reflect = itops.reflect(Deltat);

    auto Deltadlr = itops.vals2coefs(Deltat);  
    auto Deltadlr_reflect = itops.vals2coefs(Deltat_reflect);  

    if (poledlrflag == false) {
        std::cout<<"Don't do this, not implemented yet";
    }
    else {
        auto Delta_decomp = hyb_decomp(Deltadlr,dlr_rf);
        auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect,dlr_rf);
        Delta_F = hyb_F(Delta_decomp,dlr_rf, dlr_it, F, F_dag);
        Delta_F_reflect = hyb_F(Delta_decomp_reflect,dlr_rf, dlr_it, F_dag, F);
    }
    std::cout<<"Initialization done"<<std::endl;
}

nda::array<dcomplex,3> impuritysolver::Sigma_calc(nda::array_const_view<dcomplex,3> Gt, nda::array_const_view<int,2> D){
    // The sign is not included
    return Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D, Deltat, Deltat_reflect, Gt, itops, beta,  F,  F_dag);
}

nda::array<dcomplex,3> impuritysolver::G_calc(nda::array_const_view<dcomplex,3> Gt, nda::array_const_view<int,2> D){
    // The sign is not included
    return G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D, Deltat, Deltat_reflect, Gt, itops, beta,  F,  F_dag);
}