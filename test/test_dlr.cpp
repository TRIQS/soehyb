#include "../src/strong_cpl.hpp"
#include "nda/nda.hpp"
#include <cppdlr/cppdlr.hpp>
#include <gtest/gtest.h>
#include <nda/layout/policies.hpp>

using namespace cppdlr;
using namespace nda;

TEST(dyson_it, dyson_vs_ed_real) {

  // --- Problem setup --- //

  // Set problem parameters
  double beta = 10; // Inverse temperature
  const int N = 2;     // Dimension of Greens function
  int Num = 100;   // Size of equidist grid
  const int dim = 3;
  auto ID_N =  eye<dcomplex>(N);
  auto ID_dim =  eye<dcomplex>(dim);

  // Set DLR parameters
  double lambda = 10;
  double eps = 1.0e-10;

  // Get Green's function on equidist grid
  auto tgrid = nda::vector<double>(Num + 1);
  auto tgrid_relative = nda::vector<double>(Num + 1); 
  double h_t = beta / Num;
  for (int i = 0; i <= Num; ++i) {
    tgrid(i) = i * h_t;
    tgrid_relative(i) = tgrid(i)/beta;
  }
  auto Gtau = nda::array<dcomplex, 3>(Num + 1, N, N);

  double alpha_1 = 0.1;
  double alpha_2 = 0.2;

  auto G_01 = exp(-alpha_1 * tgrid);
  Gtau(range(0, Num + 1), 0, 0) = 0;
  Gtau(range(0, Num + 1), 1, 1) = 0;
  Gtau(range(0, Num + 1), 0, 1) = G_01;
  Gtau(range(0, Num + 1), 1, 0) = G_01;

  // --- Build DLR --- //

  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  // Get DLR imaginary time object
  auto itops = imtime_ops(lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  // std::cout<<dlr_rf;
  int r = itops.rank();


  auto Gdlr = itops.fitvals2coefs(tgrid_relative, Gtau);
  auto Gtau_re = nda::array<dcomplex, 3>(Num + 1, N, N);
  for (int i = 0; i <= Num; ++i) {
    Gtau_re(i, range(N), range(N)) = itops.coefs2eval(Gdlr, tgrid[i]/beta);
  }
  std::cout << "Max Gtau error: " << max_element(abs((Gtau - Gtau_re))) << std::endl;

  auto Gt = itops.coefs2vals(Gdlr);
  // auto Gt2 = nda::array<dcomplex, 3>(r, dim, dim); 
  // for (int i = 0; i <r; ++i) {
  //   Gt2(i, range(dim), range(dim)) = itops.coefs2eval(Gdlr, dlr_it(i));
  // }
  // std::cout << "Max error: " << max_element(abs((Gt - Gt2))) << std::endl; 

  auto Deltatau = nda::array<dcomplex, 3>(Num + 1, dim, dim); 
  for (int i = 0; i <= Num; ++i) Deltatau(i,_,_) = exp(-alpha_2*tgrid(i))*ID_dim;

  auto Delta_dlr = itops.fitvals2coefs(tgrid_relative, Deltatau);
  auto Deltatau_re = nda::array<dcomplex, 3>(Num + 1, dim, dim);
  for (int i = 0; i <= Num; ++i) {
    Deltatau_re(i, range(dim), range(dim)) = itops.coefs2eval(Delta_dlr, tgrid[i]/beta);
  }
  std::cout << "Max Deltatau error: " << max_element(abs((Deltatau - Deltatau_re))) << std::endl;
  auto Deltat = itops.coefs2vals(Delta_dlr);
 
  bool check = true;
  auto Delta_decomp = hyb_decomp(Delta_dlr,dlr_rf,eps,Deltat,dlr_it,check);

  auto F = nda::array<dcomplex,3>(dim,N,N);
  for (int i = 0; i<dim;++i) F(i,_,_) = ID_N;

  auto Delta_F = hyb_F(Delta_decomp, dlr_rf, dlr_it, beta, F, F);

  auto D = nda::array<int,2>{{0,2},{1,3}};
  //auto OCAdiagram = Diagram_calc(Delta_F,D,Deltat, Gt, F,  F);
  
}