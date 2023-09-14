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
    D_NCA = nda::array<int,2>{{0,1}};// Diagram topology
    D_OCA = nda::array<int,2>{{0,2},{1,3}};
    D_TCA_1 = nda::array<int,2>{{0,2},{1,4},{3,5}};
    D_TCA_2 = nda::array<int,2>{{0,3},{1,5},{2,4}};
    D_TCA_3 = nda::array<int,2>{{0,4},{1,3},{2,5}};
    D_TCA_4 = nda::array<int,2>{{0,3},{1,4},{2,5}};
    std::cout<<"Initialization done"<<std::endl;
}

nda::array<dcomplex,3> impuritysolver::Sigma_calc(nda::array_const_view<dcomplex,3> Gt, std::string order){
    nda::array<dcomplex,3> Sigma_NCA = -Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_NCA,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag);
    if ( order.compare("NCA")!=0 && order.compare("OCA")!=0 && order.compare("TCA")!=0){
        std::cout<<"order needs to be NCA, OCA or TCA"<<std::endl;
    } 
    if (order.compare("NCA")==0) return Sigma_NCA;
    else {
        nda::array<dcomplex,3> Sigma_OCA = - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_OCA,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag);
        if (order.compare("OCA")==0) return make_regular(Sigma_NCA+Sigma_OCA);
        else {
            nda::array<dcomplex,3> Sigma_TCA = -Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_1,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag)\
                            - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_2,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag)\
                            - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_3,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag)\
                            + Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_4,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag);
            if (order.compare("TCA")==0) return make_regular(Sigma_NCA+Sigma_OCA+Sigma_TCA);
        } 
    }
}

nda::array<dcomplex,3> impuritysolver::G_calc(nda::array_const_view<dcomplex,3> Gt, std::string order){
    nda::array<dcomplex,3> g_NCA = -G_Diagram_calc_sum_all(Delta_F,Delta_F_reflect,D_NCA,Deltat,Deltat_reflect, Gt,itops,beta, F,  F_dag);
    if ( order.compare("NCA")!=0 && order.compare("OCA")!=0 && order.compare("TCA")!=0){
        std::cout<<"order needs to be NCA, OCA or TCA"<<std::endl;
    } 
    if (order.compare("NCA")==0) return g_NCA;
    else {
        nda::array<dcomplex,3> g_OCA = -G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_OCA,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag); 
        if (order.compare("OCA")==0) return make_regular(g_NCA + g_OCA);
        else {
            nda::array<dcomplex,3> g_TCA = - G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_1,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag)\
                    - G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_2,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag)\
                    - G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_3,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag)\
                    + G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_4,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag);
            if (order.compare("TCA")==0) return make_regular(g_NCA + g_OCA+g_TCA);
        } 
    } 
}