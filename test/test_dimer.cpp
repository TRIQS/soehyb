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
    auto v = 1.5;
    auto tp = 1.5;

    nda::array<dcomplex,2> H = U*matmul(matmul(c0_dag,c0) ,matmul(c1_dag,c1));
    H = H- v*(matmul(c0_dag,c1)+matmul(c1_dag,c0));
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
    double lambda = beta*10;
    double eps = 1.0e-12;
    auto dlr_rf = build_dlr_rf(lambda, eps); // Get DLR frequencies
    auto itops = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
    auto const & dlr_it = itops.get_itnodes();
    int r = itops.rank();
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

    std::cout<<std::setprecision(8)<<itops.coefs2eval(G00_dlr, 0)<<std::endl;
    std::cout<<itops.coefs2eval(G01_dlr, 0)<<std::endl;

    


    

}