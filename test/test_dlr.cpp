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
  double beta = 1; // Inverse temperature
  int dim = 2;     // Dimension of Greens function
  int Num = 100;   // Size of equidist grid

  // Set DLR parameters
  double lambda = 100;
  double eps = 1.0e-14;

  // Get Green's function on equidist grid
  auto tgrid = nda::vector<double>(Num + 1);
  double h_t = beta / Num;
  for (int i = 0; i <= Num; ++i) {
    tgrid(i) = i * h_t;
  }
  auto Gtau = nda::array<dcomplex, 3>(Num + 1, dim, dim);

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
  int r = itops.rank();
  std::cout << r << std::endl;
  auto Gdlr = itops.fitvals2coefs(tgrid, Gtau);
  auto Gtau_re = nda::array<dcomplex, 3>(Num + 1, dim, dim);
  for (int i = 0; i <= Num; ++i) {
    Gtau_re(i, range(dim), range(dim)) = itops.coefs2eval(Gdlr, tgrid[i]);
  }

  std::cout << "Max error: " << max_element(abs((Gtau - Gtau_re))) << std::endl;

  auto Delta = hyb_decomp(Gdlr,dlr_rf,eps);

  std::cout<< Delta.w <<Delta.U<< Delta.V;

}