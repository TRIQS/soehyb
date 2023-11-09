#include "strong_cpl.hpp"
#include "impurity.hpp"
#include "dlr_dyson_ppsc.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <nda/linalg/matmul.hpp>

fastdiagram::fastdiagram(double beta, double lambda, double eps, nda::array<dcomplex,3> F, nda::array<dcomplex,3> F_dag):beta(beta), F(F), F_dag(F_dag){
    dlr_rf = build_dlr_rf(lambda, eps); // Get DLR frequencies
    itops = imtime_ops(lambda, dlr_rf); // construct imagninary time dlr objects 

    dlr_it = itops.get_itnodes(); //obtain imaginary time nodes
    dlr_it_actual = dlr_it;
    int r = itops.rank();
    for (int k =0;k<r;++k) {if (dlr_it_actual(k)<0) {dlr_it_actual(k) = dlr_it_actual(k)+1;}}
    dlr_it_actual = dlr_it_actual*beta;
    
    D_NCA = nda::array<int,2>{{0,1}};// NCA diagram information
    D_OCA = nda::array<int,2>{{0,2},{1,3}};// OCA diagram information
    D_TCA_1 = nda::array<int,2>{{0,2},{1,4},{3,5}}; //TCA 1st diagram information
    D_TCA_2 = nda::array<int,2>{{0,3},{1,5},{2,4}}; //TCA 2nd diagram information
    D_TCA_3 = nda::array<int,2>{{0,4},{1,3},{2,5}}; //TCA 3rd diagram information
    D_TCA_4 = nda::array<int,2>{{0,3},{1,4},{2,5}}; //TCA 4th diagram information
   // std::cout<<"Initialization done"<<std::endl;
}
nda::vector<dcomplex> fastdiagram::get_it_actual(){
    return dlr_it_actual;
}

nda::array<dcomplex,3> fastdiagram::free_greens(double beta, nda::array<dcomplex,2> H_S, double mu, bool time_order){
    return free_gf(beta, itops, H_S, mu, time_order);
}

nda::array<dcomplex,3> fastdiagram::free_greens_ppsc(double beta, nda::array<dcomplex,2> H_S){
    return free_gf_ppsc(beta, itops, H_S);
}

void fastdiagram::hyb_decomposition(nda::array<dcomplex,3> Deltat0, bool poledlrflag){
    Deltat = Deltat0;
    Deltat_reflect = itops.reflect(Deltat); // obtain Delta(-t) from Delta(t)

    auto Deltadlr = itops.vals2coefs(Deltat);  //obtain dlr coefficient of Delta(t)
    auto Deltadlr_reflect = itops.vals2coefs(Deltat_reflect); //obtain dlr coefficient of Delta(-t) 

    if (poledlrflag == false) {
        std::cout<<"Don't do this, not implemented yet";
    }
    else {
        auto Delta_decomp = hyb_decomp(Deltadlr,dlr_rf); //decomposition of Delta(t) using DLR coefficient
        auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect,dlr_rf); // decomposition of Delta(-t) using DLR coefficient
        Delta_F = hyb_F(Delta_decomp,dlr_rf, dlr_it, F, F_dag); // Compression of Delta(t) and F, F_dag matrices
        Delta_F_reflect = hyb_F(Delta_decomp_reflect,dlr_rf, dlr_it, F_dag, F);  // Compression of Delta(-t) and F, F_dag matrices
    }
}
nda::array<dcomplex,3> fastdiagram::Sigma_calc(nda::array<dcomplex,3> Gt, std::string order){

    if ( order.compare("NCA") != 0 && order.compare("OCA") != 0 && order.compare("TCA") != 0)
        throw std::runtime_error("order needs to be NCA, OCA or TCA\n");

    // First do NCA calculation
    std::cout << "S-NCA: start\n";
    nda::array<dcomplex,3> Sigma_NCA = -Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_NCA,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag);
    std::cout << "S-NCA: done\n";
    if (order.compare("NCA")==0) return Sigma_NCA;

    // Do OCA calculation
    std::cout << "S-OCA: start\n";
    nda::array<dcomplex,3> Sigma_OCA = - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_OCA,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag);
    std::cout << "S-OCA: done\n";
    if (order.compare("OCA")==0) return make_regular(Sigma_NCA+Sigma_OCA);
    
    // Do TCA calculation
    std::cout << "S-TCA: start\n";
    nda::array<dcomplex,3> Sigma_TCA =
      - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_1,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag) \
      - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_2,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag) \
      - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_3,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag) \
      + Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_4,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag);
    std::cout << "S-TCA: done\n";
    return make_regular(Sigma_NCA+Sigma_OCA+Sigma_TCA);
}

nda::array<dcomplex,3> fastdiagram::G_calc(nda::array<dcomplex,3> Gt, std::string order){

    if ( order.compare("NCA") != 0 && order.compare("OCA") != 0 && order.compare("TCA") != 0)
        throw std::runtime_error("order needs to be NCA, OCA or TCA\n");

    // First do NCA calculation
    std::cout << "G-NCA: start\n";
    nda::array<dcomplex,3> g_NCA = -G_Diagram_calc_sum_all(Delta_F,Delta_F_reflect,D_NCA,Deltat,Deltat_reflect, Gt,itops,beta, F,  F_dag);
    std::cout << "G-NCA: done\n";
    if (order.compare("NCA")==0) return g_NCA;
    
    // Do OCA calculation
    std::cout << "G-OCA: start\n";
    nda::array<dcomplex,3> g_OCA = -G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_OCA,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag); 
    std::cout << "G-OCA done\n";
    if (order.compare("OCA")==0) return make_regular(g_NCA + g_OCA);

    
    // Do TCA calculation
    std::cout << "G-TCA: start\n";
    nda::array<dcomplex,3> g_TCA =
      - G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_1,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag) \
      - G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_2,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag)	\
      - G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_3,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag)	\
      + G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_4,  Deltat, Deltat_reflect,Gt, itops,  beta,  F,  F_dag);
    std::cout << "G-TCA done\n";
    return make_regular(g_NCA + g_OCA+g_TCA);
}
nda::array<dcomplex,3> fastdiagram::time_ordered_dyson(double &beta,nda::array<dcomplex,2> H_S, double &eta_0, nda::array_const_view<dcomplex,3>Sigma_t){
    auto dys = dyson_it_ppsc(beta, itops, H_S);
    return dys.solve(Sigma_t, eta_0);
}

double fastdiagram::partition_function(nda::array<dcomplex,3> Gt){
    auto G_dlr = itops.vals2coefs(Gt);
    return -real(trace(itops.coefs2eval(G_dlr,1.0)));
}
