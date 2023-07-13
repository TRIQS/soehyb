#include "../src/strong_cpl.hpp"
#include "nda/nda.hpp"
#include <cppdlr/cppdlr.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <functional>
#include <gtest/gtest.h>
#include <nda/algorithms.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout/policies.hpp>
#include <chrono>

using namespace cppdlr;
using namespace nda;



TEST(strong_coupling, OCA_with_diag_hyb) {

  // --- Problem setup --- //

  // Set problem parameters
  double beta = 1.5; // Inverse temperature
  const int N = 2;     // Dimension of Greens function. Do not change, or you'll have to rewrite G(t)
  int Num = 128;   // Size of equidist grid
  const int dim = 3; //Dimension of hybridization. One can change dim as they want.
  auto ID_N =  eye<dcomplex>(N);
  auto ID_dim =  eye<dcomplex>(dim);

  // Set DLR parameters
  double lambda = beta;
  double eps = 1.0e-10;
  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  // Get DLR imaginary time object
  auto itops = imtime_ops(lambda, dlr_rf);
  auto const & dlr_it = itops.get_itnodes();
  int r = itops.rank();
  std::cout<< "DLR rank is "<< r<<std::endl;

  nda::vector<double> dlr_it_actual = dlr_it;
  for (int i = 0;i<r;++i) {if (dlr_it(i)<0) dlr_it_actual(i) = dlr_it(i)+1; }


  //parameters in exponential functions we will use
  double alpha_1 = 0.2;
  double alpha_2 = 0.1;
  //construct G(t) = [0, exp(-alpha_1*t); exp(-alpha_1*t) 0]
  auto Gt = nda::array<dcomplex, 3>(r, N, N);
  Gt = 0;
  auto G_01 = exp(-alpha_1*beta*dlr_it_actual);
  Gt(_,0,1) = G_01;
  Gt(_,1,0) = G_01;

  //construct Deltat = exp(-alpha2 * t)
  auto Deltat = nda::array<dcomplex, 3>(r, dim, dim); 
  Deltat = 0;
  for (int i = 0; i < r; ++i) Deltat(i,_,_) = exp(-alpha_2*beta*dlr_it_actual(i))*ID_dim;

  //construct Gdlr and Delta dlr
  auto Gdlr = itops.vals2coefs(Gt); 
  auto Deltadlr = itops.vals2coefs(Deltat);  
 
  //decomposition of hybridization
  auto Delta_decomp = hyb_decomp(Deltadlr,dlr_rf,Deltat,dlr_it,eps);
  //F matrices
  auto F = nda::array<dcomplex,3>(dim,N,N);
  for (int i = 0; i<dim;++i) F(i,_,_) = ID_N;
  
  //construct U_tilde, V_tilde, c
  auto Delta_F = hyb_F(Delta_decomp, dlr_rf, dlr_it, F, F);

  //Test OCA(2nd order) diagram
  std::cout<< "Testing OCA diagram"<<std::endl;
  auto D2 = nda::array<int,2>{{0,2},{1,3}};
  auto OCAdiagram = OCA_calc(Delta_F,Deltat, Gt,itops,beta, F,  F);
  auto begin = std::chrono::high_resolution_clock::now();
  auto OCAdiagram2 = Diagram_calc(Delta_F,D2,Deltat, Gt,itops,beta, F,  F);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t2 =  elapsed.count(); 
  std::cout<< "Difference between OCA_calc and Diagram_calc for OCA diagram is "<< max_element(abs(OCAdiagram - OCAdiagram2)) <<std::endl; 
  //EXPECT_LT(max_element(abs(OCAdiagram - OCAdiagram2)), 1e-15);
  //calculate true solution from analytic integrations
  auto OCA_true = nda::array<dcomplex,3>(Gt.shape()); 
  OCA_true = 0;
  for (int i=0;i<r;++i) OCA_true(i,0,1) =(exp(-(alpha_1+alpha_2)*beta*dlr_it_actual(i)))* (beta*dlr_it_actual(i)+(exp(-alpha_2*beta*dlr_it_actual(i))-1)/alpha_2)/alpha_2;
  OCA_true(_,1,0) = OCA_true(_,0,1);
  OCA_true = OCA_true * pow(dim,2);

  std::cout<< "Error of OCA_Diagram is "<< max_element(abs(OCAdiagram - OCA_true)) <<std::endl; 
  //EXPECT_LT(max_element(abs(OCAdiagram - OCA_true)), 1e-12);

  

  //Test third order diagram
  auto D3 = nda::array<int,2>{{0,2},{1,4},{3,5}};
  std::cout<< "Testing third order diagram with topology: "<<D3<<std::endl; 
  begin = std::chrono::high_resolution_clock::now();
  auto diagram_3rd_order = Diagram_calc(Delta_F,D3,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t3 =  elapsed.count();
  //construct analytic solution
  auto diagram_true = nda::array<dcomplex,3>(Gt.shape());  
  diagram_true = 0;
  for (int i=0;i<r;++i) diagram_true(i,0,1) = (1/(2*pow(alpha_2,4))) *exp(-(alpha_1+alpha_2)*beta*dlr_it_actual(i))*(pow(alpha_2,2)*pow(beta*dlr_it_actual(i),2)-4*alpha_2*beta*dlr_it_actual(i)-2*exp(-alpha_2*beta*dlr_it_actual(i))*(alpha_2*beta*dlr_it_actual(i)+3)+6);

  diagram_true(_,1,0) = diagram_true(_,0,1);
  diagram_true = diagram_true * pow(dim,D3.shape(0));
  std::cout<<"Error of 3rd order diagram is "<<max_element(abs(diagram_true - diagram_3rd_order)) <<std::endl;

  // Calculate  fourth and fifth order diagram:
  auto D4= nda::array<int,2>{{0,2},{1,6},{3,5},{4,7}};
  begin = std::chrono::high_resolution_clock::now();
  auto diagram_4th_order = Diagram_calc(Delta_F,D4,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t4 =  elapsed.count();

  auto D5= nda::array<int,2>{{0,2},{1,8},{3,6},{4,9},{5,7}};
  // begin = std::chrono::high_resolution_clock::now();
  // auto diagram_5th_order = Diagram_calc(Delta_F,D5,Deltat, Gt,itops,beta, F,  F);
  // end = std::chrono::high_resolution_clock::now();
  // elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  // double t5 =  elapsed.count(); 

  std::cout<< "Time spent for calculating diagrams for m = 2,3,4 is "<<" "<< t2 << " "<<t3<<" "<<t4<<std::endl;
  int P = Delta_decomp.U.shape(0);
  std::cout<<"Theoretical complexity P^(m-1)*m for m=2,3,4,5 is ";
  for (int m=2;m<=5;++m) std::cout<< pow(P,m-1)*m<<" ";
  std::cout<<std::endl<< "Rescaled both of them:"<< std::endl;
  std::cout<<"Time actually spent,    divided by "<< t2<<" is "<<t2/t2<<" "<<int(t3/t2)<<" "<<int(t4/t2)<<std::endl;
  std::cout<<"Theoretical complexity, divided by "<< pow(P,2-1)*2<<" is "<<pow(P,2-1)*2/(pow(P,2-1)*2)<<" "<<pow(P,3-1)*3/(pow(P,2-1)*2)<<" "<<pow(P,4-1)*4/(pow(P,2-1)*2)<<" "<<std::endl;
  
  std::cout<<"testing much fewer poles"<<std::endl;
  nda::array<double,1> pol(1);
  pol(0) =  alpha_2*beta;
  nda::array<dcomplex,3> A(1,dim,dim);
  A(0,_,_) = ID_dim/k_it(0,alpha_2*beta);
  auto Delta_decomp_simple = hyb_decomp(A,pol,Deltat,dlr_it,eps);
  auto Delta_F_simple = hyb_F(Delta_decomp_simple, dlr_rf, dlr_it, F, F);
  begin = std::chrono::high_resolution_clock::now();
  auto OCAdiagram_simple = Diagram_calc(Delta_F_simple,D2,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t2_simple =  elapsed.count(); 
  std::cout<< "Error of simple OCA_Diagram is "<< max_element(abs(OCAdiagram_simple - OCA_true)) <<std::endl; 
  begin = std::chrono::high_resolution_clock::now();
  auto diagram_3rd_order_simple = Diagram_calc(Delta_F_simple,D3,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t3_simple =  elapsed.count();
  begin = std::chrono::high_resolution_clock::now();
  auto diagram_4th_order_simple = Diagram_calc(Delta_F_simple,D4,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t4_simple =  elapsed.count(); 
  begin = std::chrono::high_resolution_clock::now();
  auto diagram_5th_order_simple = Diagram_calc(Delta_F_simple,D5,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t5_simple =  elapsed.count(); 
  begin = std::chrono::high_resolution_clock::now();
  auto D6 = nda::array<int,2>{{0,2},{1,7},{3,9},{4,10},{5,8},{6,11}}; 
  auto diagram_6th_order_simple = Diagram_calc(Delta_F_simple,D6,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t6_simple =  elapsed.count(); 
  auto D7 = nda::array<int,2>{{0,2},{1,7},{3,13},{4,10},{5,8},{6,11},{9,12}}; 
  auto diagram_7th_order_simple = Diagram_calc(Delta_F_simple,D7,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t7_simple =  elapsed.count(); 
  auto D8 = nda::array<int,2>{{0,2},{1,14},{3,13},{4,10},{5,8},{6,11},{9,12},{7,15}}; 
  auto diagram_8th_order_simple = Diagram_calc(Delta_F_simple,D8,Deltat, Gt,itops,beta, F,  F);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t8_simple =  elapsed.count();  
  
 
  std::cout<<"Time comparison in seconds"<<std::endl;
  std::cout<<"      DLR poles            "<<"Simple poles"<<std::endl;
  std::cout<<"m=2      "<< t2/1000<<"                    "<<t2_simple/1000<<std::endl;
  std::cout<<"m=3      "<< t3/1000<<"                   "<<t3_simple/1000<<std::endl;
  std::cout<<"m=4      "<< t4/1000<<"                 "<<t4_simple/1000<<std::endl;
  std::cout<<"m=5      "<< t2*pow(P,5-1)*5/(pow(P,2-1)*2)/1000<<"(prediction)     "<<t5_simple/1000<<std::endl;
  std::cout<<"m=6      "<< t2*pow(P,6-1)*6/(pow(P,2-1)*2)/1000<<"(prediction)     "<<t6_simple/1000<<std::endl; 
  std::cout<<"m=7      "<< t2*pow(P,7-1)*7/(pow(P,2-1)*2)/1000<<"(prediction)     "<<t7_simple/1000<<std::endl;
  std::cout<<"m=8      "<< t2*pow(P,8-1)*8/(pow(P,2-1)*2)/1000<<"(prediction)     "<<t8_simple/1000<<std::endl; 
 
}
