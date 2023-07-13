#include "../src/strong_cpl.hpp"
#include "nda/nda.hpp"
#include <cppdlr/cppdlr.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <functional>
#include <gtest/gtest.h>
#include <nda/algorithms.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout/policies.hpp>

using namespace cppdlr;
using namespace nda;



TEST(strong_coupling, OCA_with_diag_hyb) {

  // --- Problem setup --- //

  // Set problem parameters
  double beta = 1; // Inverse temperature
  const int N = 2;     // Dimension of Greens function. Do not change, or you'll have to rewrite G(t)
  int Num = 128;   // Size of equidist grid
  const int dim = 3; //Dimension of hybridization. One can change dim as they want.
  auto ID_N =  eye<dcomplex>(N);
  auto ID_dim =  eye<dcomplex>(dim);

  // Set DLR parameters
  double lambda = beta;
  double eps = 1.0e-11;
  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  // Get DLR imaginary time object
  auto itops = imtime_ops(lambda, dlr_rf);
  auto const & dlr_it = itops.get_itnodes();
  int r = itops.rank();

  nda::vector<double> dlr_it_actual = dlr_it;
  for (int i = 0;i<r;++i) {if (dlr_it(i)<0) dlr_it_actual(i) = dlr_it(i)+1; }


  //parameters in exponential functions we will use
  double alpha_1 = 0.2;
  double alpha_2 = 0.1;
  //construct G(t) = [0, exp(-alpha_1*t); exp(-alpha_1*t) 0]
  auto Gt = nda::array<dcomplex, 3>(r, N, N);
  Gt = 0;
  auto G_01 = exp(-alpha_1*dlr_it_actual);
  Gt(_,0,1) = G_01;
  Gt(_,1,0) = G_01;

  //construct Deltat = exp(-alpha2 * t)
  auto Deltat = nda::array<dcomplex, 3>(r, dim, dim); 
  Deltat = 0;
  for (int i = 0; i < r; ++i) Deltat(i,_,_) = exp(-alpha_2*dlr_it_actual(i))*ID_dim;

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
  auto OCAdiagram2 = Diagram_calc(Delta_F,D2,Deltat, Gt,itops,beta, F,  F); 
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
  auto diagram_3rd_order = Diagram_calc(Delta_F,D3,Deltat, Gt,itops,beta, F,  F);
  //construct analytic solution
  auto diagram_true = nda::array<dcomplex,3>(Gt.shape());  
  diagram_true = 0;
  for (int i=0;i<r;++i) diagram_true(i,0,1) = (1/(2*pow(alpha_2,4))) *exp(-(alpha_1+alpha_2)*beta*dlr_it_actual(i))*(pow(alpha_2,2)*pow(beta*dlr_it_actual(i),2)-4*alpha_2*beta*dlr_it_actual(i)-2*exp(-alpha_2*beta*dlr_it_actual(i))*(alpha_2*beta*dlr_it_actual(i)+3)+6);

  diagram_true(_,1,0) = diagram_true(_,0,1);
  diagram_true = diagram_true * pow(dim,D3.shape(0));
  std::cout<<"Error of oca diagram is "<<max_element(abs(diagram_true - diagram_3rd_order)) <<std::endl;
}
