#include "../src/strong_cpl.hpp"
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

nda::array<dcomplex,3> ppsc_free_greens_tau(nda::vector_const_view<double> tau_i, nda::array_view<dcomplex,2> H_S, double beta);
nda::array<dcomplex,3> NCA(nda::array_view<dcomplex,3> Deltat,nda::array_view<dcomplex,3> Deltat_reflect,nda::array<dcomplex,3> G_iaa,nda::array_const_view<dcomplex,3> F,nda::array_const_view<dcomplex,3> F_dag);
// nda::array<dcomplex,3> step(nda::array_view<dcomplex,2> G_iaa);
TEST(strong_coupling, dimer) {
    int N = 64;
    auto c0_dag = nda::array<dcomplex,2>(N,N); c0_dag = 0;
    auto c1_dag = nda::array<dcomplex,2>(N,N); c1_dag = 0;
    auto c2_dag = nda::array<dcomplex,2>(N,N); c2_dag = 0;//b00
    auto c3_dag = nda::array<dcomplex,2>(N,N); c3_dag = 0;//b01
    auto c4_dag = nda::array<dcomplex,2>(N,N); c4_dag = 0;//b01
    auto c5_dag = nda::array<dcomplex,2>(N,N); c5_dag = 0;//b11

    for (int i = 0;i<32;++i) c0_dag(i+32,i) = 1;

    for (int i = 0;i<16;++i) c1_dag(i+16,i) = 1;
    for (int i = 32;i<48;++i) c1_dag(i+16,i) = -1;

    for (int i = 0;i<8;++i) {
        c2_dag(i+8,i) = 1; c2_dag(i+24,i+16) = -1;  
        c2_dag(i+40,i+32) = -1; c2_dag(i+56,i+48) = 1;  
    }
    for (int i=0;i<4;++i){
        c3_dag(i+4,i) = 1; c3_dag(i+12,i+8) = -1; c3_dag(i+20,i+16) = -1; c3_dag(i+28,i+24) = 1; 
        c3_dag(i+36,i+32) = -1; c3_dag(i+44,i+40) = 1; c3_dag(i+52,i+48) = 1; c3_dag(i+60,i+56) = -1;   
    }

    for (int i = 0;i<16;++i) {
        if ((i==1) or (i==2) or (i==4) or (i==8) or (i==14) or (i==13) or (i==11) or (i==7)){
            c4_dag(4*i+2,4*i) = -1; c4_dag(4*i+3,4*i+1) = -1;
        }
        else {c4_dag(4*i+2,4*i) = 1; c4_dag(4*i+3,4*i+1) = 1;}
    }
    for (int i=0;i<32;++i){
       if ((i==1) or (i==2) or (i==4) or (i==8) or (i==14) or (i==13) or (i==11) or (i==7)) c5_dag(2*i+1,2*i) = -1; 
       else if ((i==16) or (i==19) or (i==21) or (i==25) or (i==22) or (i==26) or (i==28) or (i==31)) c5_dag(2*i+1,2*i) = -1; 
       else c5_dag(2*i+1,2*i) = 1;  
    }
    

    auto c0 = conj(transpose(c0_dag)); auto c1 = conj(transpose(c1_dag)); auto c2 = conj(transpose(c2_dag));
    auto c3 = conj(transpose(c3_dag)); auto c4 = conj(transpose(c4_dag)); auto c5 = conj(transpose(c5_dag));
    // for (int i =0;i<64;++i) std::cout<<(matmul(c0,c1_dag)+matmul(c1_dag,c0))(i,i);

    double t = 1.0;
    auto U=4*t;
    auto v = 0.0;
    auto tp = 1.5;
    auto mu = 2.0;

    nda::array<dcomplex,2> H = U*matmul(matmul(c0_dag,c0) ,matmul(c1_dag,c1));
    H = H- v*(matmul(c0_dag,c1)+matmul(c1_dag,c0));
    H = H - mu*(matmul(c0_dag,c0)+matmul(c1_dag,c1));
    H = H - t*(matmul(c0_dag,c2)+matmul(c2_dag,c0))- t*(matmul(c0_dag,c3)+matmul(c3_dag,c0));
    H = H - t*(matmul(c1_dag,c4)+matmul(c4_dag,c1))- t*(matmul(c1_dag,c5)+matmul(c5_dag,c1));
    H = H - tp*(matmul(c2_dag,c4)+matmul(c4_dag,c2))- tp*(matmul(c3_dag,c5)+matmul(c5_dag,c3));

    
    auto [eval, evec] = nda::linalg::eigenelements(H);
   
    
    // std::cout<<exp_eval;
    double beta = 32;
    nda::vector<double> exp_eval = exp(-beta*(eval-min_element(eval)));
    exp_eval = exp_eval/sum(exp_eval); 
    double Z = sum(exp(-beta*eval));
   // std::cout<<((eval));
    double lambda = 1000;
    double eps = 1.0e-15;
    auto dlr_rf = build_dlr_rf(lambda, eps); // Get DLR frequencies
    auto itops = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
    auto const & dlr_it = itops.get_itnodes();
    int r = itops.rank();
    std::cout<<"dlr rank is"<< r;
    nda::vector<double> tau_actual = itops.get_itnodes();
    for (int k =0;k<r;++k){
        if (tau_actual(k)<0) {tau_actual(k) = tau_actual(k)+1;}
    }
    tau_actual = tau_actual*beta;
    
    int n_all=6;

    int N_t = 1000;
    auto t_relative = nda::vector<double>(N_t);
    for (int i=0;i<N_t;++i) {t_relative(i) = (i+0.0)/N_t; if (t_relative(i)>0.5) {t_relative(i) -= 1;};}

    auto G00 = nda::array<dcomplex,3>(r,1,1); G00=0;
    auto G01 = nda::array<dcomplex,3>(r,1,1); G01=0;

    
    auto V_sr_0 = matmul(conj(transpose(evec)),matmul(c0,evec));
    
    
    auto V_sr_1 = matmul(conj(transpose(evec)),matmul(c1,evec));
    for (int s=0;s<N;++s){
        for (int s2=0;s2<N;++s2){
            //for (int k=0;k<N_t;++k){
            for (int k=0;k<r;++k){ 
                double w = eval(s2)-eval(s);
                G00(k,0,0) += V_sr_0(s,s2)* conj(V_sr_0(s,s2))*(exp_eval(s)+exp_eval(s2))*k_it(dlr_it(k),w*beta);
                G01(k,0,0) += V_sr_0(s,s2)* conj(V_sr_1(s,s2))*(exp_eval(s)+exp_eval(s2))*k_it(dlr_it(k),w*beta);
            }
        }
    }

    auto G00_dlr = itops.vals2coefs(G00);
    auto G01_dlr = itops.vals2coefs(G01);

    // std::cout<<std::setprecision(8)<<itops.coefs2eval(G00_dlr, 0)<<std::endl;
    // std::cout<<itops.coefs2eval(G01_dlr, 0)<<std::endl;

    auto h_bath = nda::array<dcomplex,2>(2,2);h_bath=0;
    h_bath(0,1) = -tp; h_bath(1,0) = -tp;
    auto Deltat = make_regular(2* free_gf(beta, itops,h_bath,0)* pow(t,2));

    int N_S = 4;
    auto c0_S_dag = nda::array<dcomplex,2>(N_S,N_S); c0_dag = 0;
    auto c1_S_dag = nda::array<dcomplex,2>(N_S,N_S); c1_dag = 0;
    c0_S_dag(2,0) = 1; c0_S_dag(3,1) = 1;
    c1_S_dag(1,0) = 1; c1_S_dag(3,2) = -1; 
    auto c0_S = conj(transpose(c0_S_dag));
    auto c1_S = conj(transpose(c1_S_dag)); 
    
    auto F = nda::array<dcomplex,3>(2,N_S,N_S); F(0,_,_) = c0_S; F(1,_,_) = c1_S;
    auto F_dag = nda::array<dcomplex,3>(2,N_S,N_S);F_dag(0,_,_) = c0_S_dag; F_dag(1,_,_) = c1_S_dag;


    nda::array<dcomplex,2> H_S = U*matmul(matmul(c0_S_dag,c0_S) ,matmul(c1_S_dag,c1_S));
    H_S = H_S- v*(matmul(c0_S_dag,c1_S)+matmul(c1_S_dag,c0_S)); 
    H_S = H_S - mu*(matmul(c0_S_dag,c0_S)+matmul(c1_S_dag,c1_S));
    
    auto G0_S_tau = ppsc_free_greens_tau(tau_actual, H_S, beta);

    auto G0_S_dlr = itops.vals2coefs(G0_S_tau);
    auto Deltat_reflect = itops.reflect(Deltat);
    auto Sigma_NCA = NCA(Deltat,Deltat_reflect, G0_S_tau,F,F_dag);
    

    auto Deltadlr = itops.vals2coefs(Deltat);  
    auto Deltadlr_reflect = itops.vals2coefs(Deltat_reflect);
   // auto Delta_decomp = hyb_decomp(Deltadlr,dlr_rf);
   // auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect,dlr_rf);
    auto Delta_M = nda::array<dcomplex,3>{{{0.5,0.5},{0.5,0.5}},{{0.5,-0.5},{-0.5,0.5}}}; 
    Delta_M =Delta_M*2*t*t;
    auto Delta_p = nda::vector<double> {-tp,tp}; 
    Delta_p *= beta;
    nda::vector<double> Delta_p_reflect = -Delta_p;
    auto Delta_decomp = hyb_decomp(Delta_M,Delta_p);
    auto Delta_decomp_reflect = hyb_decomp(Delta_M,Delta_p_reflect);
    Delta_decomp.check_accuracy(Deltat, dlr_it);
    Delta_decomp_reflect.check_accuracy(Deltat_reflect, dlr_it);
    auto Delta_F = hyb_F(Delta_decomp,dlr_rf, dlr_it, F, F_dag);
    auto Delta_F_reflect = hyb_F(Delta_decomp_reflect,dlr_rf, dlr_it, F_dag, F);
    double eta;
    double Z_S;
    
    auto G_S_tau = G0_S_tau;
    auto D_NCA = nda::array<int,2>{{0,1}};// Diagram topology
    auto D2 = nda::array<int,2>{{0,2},{1,3}};
    nda::array<dcomplex,3> G_S_tau_old = 0.0*G_S_tau; 
    for (int ppsc_iter = 0; ppsc_iter<100;++ppsc_iter){
       
        if (max_element(abs(G_S_tau_old-G_S_tau))<1e-10) break;
        auto NCAdiagram = -Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_NCA,  Deltat, Deltat_reflect,G_S_tau, itops,  beta,  F,  F_dag);
    
        auto OCAdiagram = -Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D2,  Deltat, Deltat_reflect,G_S_tau, itops,  beta,  F,  F_dag);
        auto Sigma_t = make_regular(NCAdiagram+OCAdiagram);
      //  std::cout<<"Sigma(t) is"<<Sigma_t(0,_,_)<<std::endl;
        auto Sigma_dlr = itops.vals2coefs(Sigma_t);//));
    
        auto dys = dyson_it(beta, itops, H_S, 0, true);
        auto G_new_tau   = dys.solve(Sigma_t);  
        auto G_new_dlr = itops.vals2coefs(G_new_tau);
      //  std::cout<<itops.coefs2eval(G_new_dlr, 0.0/32.0)<<std::endl;
       // std::cout<<itops.coefs2eval(G_new_dlr, 3.0/32.0);
        Z_S= -real(trace(itops.coefs2eval(G_new_dlr,1.0)));//should be changed
        eta = log(Z_S)/beta;
       // std::cout<<Z_S;
        H_S += eta*nda::eye<dcomplex>(H_S.shape(0));
        G_S_tau_old = G_S_tau;
        for (int k=0;k<r;++k){
            G_new_tau(k,_,_) =  G_new_tau(k,_,_) * exp(-tau_actual(k)*eta);
        }
        G_S_tau = 0.5*G_S_tau + 0.5* G_new_tau;
         std::cout<<"iter "<<ppsc_iter<<" , diff is "<<max_element(abs(G_S_tau_old-G_S_tau))<<std::endl;
    }
    auto G_S_dlr = itops.vals2coefs(G_S_tau);
    std::cout<<itops.coefs2eval(G_S_dlr,1.33817298e-04/32.0); 
    std::cout<<itops.coefs2eval(G_S_dlr,5.03068549e-02/32.0);
    std::cout<<itops.coefs2eval(G_S_dlr,2.67856053e+00/32.0);
    auto g_S_NCA = -G_Diagram_calc_sum_all(Delta_F,Delta_F_reflect,D_NCA,Deltat,Deltat_reflect, G_S_tau,itops,beta, F,  F_dag);
    std::cout<<make_regular(g_S_NCA(0,_,_));
    auto g_S_OCA = -G_Diagram_calc_sum_all(Delta_F,Delta_F_reflect,D2,Deltat,Deltat_reflect, G_S_tau,itops,beta, F,  F_dag);
    auto g_S_OCA_dlr = itops.vals2coefs(make_regular(g_S_OCA));
    std::cout<<itops.coefs2eval(g_S_OCA_dlr,1.33817298e-04/32.0); 
    std::cout<<itops.coefs2eval(g_S_OCA_dlr,5.03068549e-02/32.0);
    std::cout<<itops.coefs2eval(g_S_OCA_dlr,2.67856053e+00/32.0)<<std::endl;


    for (int i=0;i<r;++i) std::cout<<abs(G00(i,0,0)-g_S_NCA(i,0,0))<<" ";
    std::cout<<std::endl;
    for (int i=0;i<r;++i) std::cout<<abs(G00(i,0,0)-g_S_NCA(i,0,0)-2*g_S_OCA(i,0,0))<<" "; 
}

nda::array<dcomplex,3> ppsc_free_greens_tau(nda::vector_const_view<double> tau_i, nda::array_view<dcomplex,2> H_S, double beta){
    int na = H_S.shape(0);
    nda::array<dcomplex,2> I_aa = nda::eye<dcomplex>(na);
    auto [E,U] = nda::linalg::eigenelements(H_S);

    auto E0 = min_element(E);
    E-=E0;
    auto Z = sum(exp(-beta*E));
    auto eta = log(Z)/beta;
    E+= eta;

    int r = tau_i.shape(0);

    auto g_iaa = nda::array<dcomplex,3>(r,na,na);
    auto exp_ia = nda::array<dcomplex,2>(r,na);
    for (int k=0;k<r;++k) exp_ia(k,_) = exp(-tau_i(k)*E);

    
    for (int k=0;k<r;++k){
        g_iaa(k,_,_) = -matmul(U,matmul(diag(exp_ia(k,_)),conj(transpose(U))));
    }
    H_S += I_aa*(eta-E0);
    return g_iaa;
}

// nda::array<dcomplex,3> step(imtime_ops &itops, nda::array_view<dcomplex,3> Deltat,nda::array_view<dcomplex,3> G_iaa,nda::array_view<dcomplex,2>H ,nda::vector_const_view<double> tau_actual,nda::vector_const_view<double> dlr_rf, nda::array_const_view<dcomplex,3> F,nda::array_const_view<dcomplex,3> F_dag,double beta){
//     auto Deltat_reflect = itops.reflect(Deltat);
//     auto Sigma_iaa = NCA(Deltat,Deltat_reflect,G_iaa,F,F_dag);
//     //higher orders to be put here ....
//     auto Sigma_xaa = itops.vals2coefs(Sigma_iaa);
//     auto G_xaa_new = dyson_dlr_integrodiff(itops,H, Sigma_xaa,dlr_rf, beta);
//     auto Z = partition_func(G_xaa_new, beta);
//     double eta = log(Z)/beta;
//     H += eta*nda::eye<dcomplex>(H.shape(0));
//     auto G_iaa_new = itops.coefs2vals(G_xaa_new);
//     int r = tau_actual.shape(0);
//     for (int k=0;k<r;++k){
//         G_iaa_new(k,_,_) =  G_iaa_new(k,_,_) * exp(-tau_actual(k)*eta);
//     }
//     return G_iaa_new; 
// }



nda::array<dcomplex,3> NCA(nda::array_view<dcomplex,3> Deltat,nda::array_view<dcomplex,3> Deltat_reflect,nda::array<dcomplex,3> G_iaa,nda::array_const_view<dcomplex,3> F,nda::array_const_view<dcomplex,3> F_dag){
    auto Sigma_iaa = nda::array<dcomplex,3>(G_iaa.shape());
    Sigma_iaa = 0;
    int n = F.shape(0);
    int r = G_iaa.shape(0);
    for (int k=0;k<r;++k){
        for (int n1=0;n1<n;++n1) {
            for (int n2=0;n2<n;++n2){
                Sigma_iaa(k,_,_) -= Deltat(k,n1,n2)*(matmul(F_dag(n1,_,_), matmul(G_iaa(k,_,_),F(n2,_,_))));
                Sigma_iaa(k,_,_) -= Deltat_reflect(k,n1,n2)*(matmul(F(n1,_,_), matmul(G_iaa(k,_,_),F_dag(n2,_,_))));
            }
        }
    }
    return Sigma_iaa;
}