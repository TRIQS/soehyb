#include <chrono>
#include <cppdlr/dlr_imfreq.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <cppdlr/utils.hpp>
#include <h5/complex.hpp>
#include <h5/generic.hpp>
#include <h5/object.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nda/algorithms.hpp>
#include <nda/basic_functions.hpp>
#include <nda/declarations.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/linalg/matmul.hpp>
#include <nda/mapped_functions.hxx>
#include <nda/nda.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_manual.hpp>
#include <triqs_soehyb/backbone.hpp>
#include <triqs_soehyb/dense_backbone.hpp>
#include <triqs_soehyb/block_sparse_backbone.hpp>
#include <gtest/gtest.h>

using namespace nda;

// NOTE: Tests with PYTHON_ in the name load h5 files and should probably be performed on the Python side

nda::array<dcomplex, 3> Hmat_to_Gtmat(nda::array<dcomplex, 2> Hmat, double beta, nda::array<double, 1> dlr_it_abs) {
  // Helper function for computing the non-interacting Green's function from the Hamiltonian, both in dense storage

  int N                         = Hmat.extent(0);
  auto [H_loc_eval, H_loc_evec] = nda::linalg::eigenelements(Hmat);
  auto E0                       = nda::min_element(H_loc_eval);
  H_loc_eval -= E0;
  auto tr_exp_minusbetaH = nda::sum(exp(-beta * H_loc_eval));
  auto eta_0             = nda::log(tr_exp_minusbetaH) / beta;
  H_loc_eval += eta_0;
  auto Gt_evals_t = nda::zeros<dcomplex>(N, N);
  int r           = dlr_it_abs.extent(0);
  auto Gt_mat     = nda::zeros<dcomplex>(r, N, N);
  auto Gbeta      = nda::zeros<dcomplex>(N, N);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < N; i++) { Gt_evals_t(i, i) = -exp(-beta * dlr_it_abs(t) * H_loc_eval(i)); }
    Gt_mat(t, _, _) = nda::matmul(H_loc_evec, nda::matmul(Gt_evals_t, nda::transpose(H_loc_evec)));
  }
  return Gt_mat;
}

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> discrete_bath_helper(double beta, double Lambda, double eps) {
  // Helper function for setting up the discrete bath model

  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // hybridization parameters
  double s = 0.5;
  double t = 1.0;
  nda::array<double, 1> e{-2.3 * t, 2.3 * t};

  // hybridization generation
  auto Jt      = nda::array<dcomplex, 3>(r, 1, 1);
  auto Jt_refl = nda::array<dcomplex, 3>(r, 1, 1);
  for (int i = 0; i <= 1; i++) {
    for (int u = 0; u < r; u++) {
      Jt(u, 0, 0) += k_it(dlr_it_abs(u), e(i), beta);
      Jt_refl(u, 0, 0) += k_it(-dlr_it_abs(u), e(i), beta);
    }
  }

  // orbital index order: do 0, do 1, up 0, up 1. same level <-> same parity index
  auto Deltat      = nda::array<dcomplex, 3>(r, 4, 4);
  auto Deltat_refl = nda::array<dcomplex, 3>(r, 4, 4);

  for (int i = 0; i < Deltat.extent(1); i++) {
    for (int j = i; j < Deltat.extent(2); j++) {
      if (i == j) {
        Deltat(_, i, j)      = Jt(_, 0, 0);
        Deltat_refl(_, i, j) = Jt_refl(_, 0, 0);
      } else if ((i == 0 && j == 1) || (i == 1 && j == 0) || (i == 2 && j == 3) || (i == 3 && j == 2)) {
        Deltat(_, i, j)      = s * Jt(_, 0, 0);
        Deltat_refl(_, i, j) = s * Jt_refl(_, 0, 0);
      }
    }
  }
  Deltat      = t * t * Deltat;
  Deltat_refl = t * t * Deltat_refl;

  return std::make_tuple(Deltat, Deltat_refl);
}

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> discrete_bath_spin_flip_helper(double beta, double Lambda, double eps, int n) {
  // Helper function for setting up the discrete bath model

  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // hybridization parameters
  double s = 0.5;
  double t = 1.0;
  nda::array<double, 1> e{-2.3 * t, 2.3 * t};

  // hybridization generation
  auto Jt      = nda::array<dcomplex, 3>(r, 1, 1);
  auto Jt_refl = nda::array<dcomplex, 3>(r, 1, 1);
  for (int i = 0; i <= 1; i++) {
    for (int u = 0; u < r; u++) {
      Jt(u, 0, 0) += k_it(dlr_it_abs(u), e(i), beta);
      Jt_refl(u, 0, 0) += k_it(-dlr_it_abs(u), e(i), beta);
    }
  }

  // orbital index order: do 0, up 0, do 1, up 1
  auto Deltat      = nda::array<dcomplex, 3>(r, n, n);
  auto Deltat_refl = nda::array<dcomplex, 3>(r, n, n);

  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      if (i == j) {
        Deltat(_, i, j)      = Jt(_, 0, 0);
        Deltat_refl(_, i, j) = Jt_refl(_, 0, 0);
      } else if ((i + j) % 2 == 0) { // i and j have the same parity
        Deltat(_, i, j)      = s * Jt(_, 0, 0);
        Deltat_refl(_, i, j) = s * Jt_refl(_, 0, 0);
      }
    }
  }
  Deltat      = t * t * Deltat;
  Deltat_refl = t * t * Deltat_refl;

  return std::make_tuple(Deltat, Deltat_refl);
}

std::tuple<int, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, BlockDiagOpFun, std::vector<BlockOp>, std::vector<BlockOp>, nda::array<dcomplex, 3>,
           nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, std::vector<std::vector<unsigned long>>, std::vector<long>>
two_band_discrete_bath_helper(double beta, double Lambda, double eps) {
  // Helper function for setting up the two-band discrete bath model

  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // hybridization parameters
  double s = 0.5;
  double t = 1.0;
  nda::array<double, 1> e{-2.3 * t, 2.3 * t};

  // hybridization generation
  auto Jt      = nda::array<dcomplex, 3>(r, 1, 1);
  auto Jt_refl = nda::array<dcomplex, 3>(r, 1, 1);
  for (int i = 0; i <= 1; i++) {
    for (int u = 0; u < r; u++) {
      Jt(u, 0, 0) += k_it(dlr_it_abs(u), e(i), beta);
      Jt_refl(u, 0, 0) += k_it(-dlr_it_abs(u), e(i), beta);
    }
  }
  // orbital index order: do 0, do 1, up 0, up 1. same level <-> same parity index
  auto Deltat      = nda::array<dcomplex, 3>(r, 4, 4);
  auto Deltat_refl = nda::array<dcomplex, 3>(r, 4, 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j) {
        Deltat(_, i, j)      = Jt(_, 0, 0);
        Deltat_refl(_, i, j) = Jt_refl(_, 0, 0);
      } else if ((i == 0 && j == 1) || (i == 1 && j == 0) || (i == 2 && j == 3) || (i == 3 && j == 2)) {
        Deltat(_, i, j)      = s * Jt(_, 0, 0);
        Deltat_refl(_, i, j) = s * Jt_refl(_, 0, 0);
      }
    }
  }
  Deltat      = t * t * Deltat;
  Deltat_refl = t * t * Deltat_refl;

  // get Hamiltonian, creation/annihilation operators in block-sparse storage
  int num_blocks = 5; // number of blocks of Hamiltonian

  // Hamiltonian
  std::vector<nda::array<double, 2>> H_blocks(num_blocks); // Hamiltonian in sparse storage
  H_blocks[0]                   = nda::make_regular(-1 * nda::eye<double>(4));
  H_blocks[1]                   = {{-0.6, 0, 0, 0, 0, 0},   {0, 8.27955e-19, 0, 0, 0.2, 0}, {0, 0, -0.4, 0.2, 0, 0},
                                   {0, 0, 0.2, -0.4, 0, 0}, {0, 0.2, 0, 0, 8.27955e-19, 0}, {0, 0, 0, 0, 0, -0.6}};
  H_blocks[2]                   = {{0}};
  H_blocks[3]                   = nda::make_regular(2 * nda::eye<double>(4));
  H_blocks[4]                   = {{6}};
  nda::vector<int> H_block_inds = {0, 0, -1, 0, 0};

  // Green's function
  auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);

  // creation/annihilation operators
  nda::vector<int> ann_conn = {2, 0, -1, 1, 3}; // block column indices of F operators
  nda::vector<int> cre_conn = {1, 3, 0, 4, -1}; // block column indices of F^dag operators
  std::vector<BlockOp> Fs;
  std::vector<BlockOp> Fdags;
  std::vector<nda::array<dcomplex, 2>> dummy(num_blocks);
  std::vector<std::vector<nda::array<dcomplex, 2>>> F_blocks(4, dummy);
  std::vector<std::vector<nda::array<dcomplex, 2>>> Fdag_blocks(4, dummy);

  F_blocks[0][0] = {{1, 0, 0, 0}};
  F_blocks[0][1] = {{0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0}};
  F_blocks[0][2] = {{0}};
  F_blocks[0][3] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
  F_blocks[0][4] = {{0}, {0}, {0}, {1}};

  F_blocks[1][0] = {{0, 1, 0, 0}};
  F_blocks[1][1] = {{-1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}};
  F_blocks[1][2] = {{0}};
  F_blocks[1][3] = {{0, 0, 0, 0}, {-1, 0, 0, 0}, {0, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};
  F_blocks[1][4] = {{0}, {0}, {-1}, {0}};

  F_blocks[2][0] = {{0, 0, 1, 0}};
  F_blocks[2][1] = {{0, -1, 0, 0, 0, 0}, {0, 0, -1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}};
  F_blocks[2][2] = {{0}};
  F_blocks[2][3] = {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, -1}, {0, 0, 0, 0}};
  F_blocks[2][4] = {{0}, {1}, {0}, {0}};

  F_blocks[3][0] = {{0, 0, 0, 1}};
  F_blocks[3][1] = {{0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, -1, 0}, {0, 0, 0, 0, 0, -1}, {0, 0, 0, 0, 0, 0}};
  F_blocks[3][2] = {{0}};
  F_blocks[3][3] = {{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  F_blocks[3][4] = {{-1}, {0}, {0}, {0}};

  Fdag_blocks[0][0] = {{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  Fdag_blocks[0][1] = {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0}};
  Fdag_blocks[0][2] = {{1}, {0}, {0}, {0}};
  Fdag_blocks[0][3] = {{0, 0, 0, 1}};
  Fdag_blocks[0][4] = {{0}};

  Fdag_blocks[1][0] = {{-1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}};
  Fdag_blocks[1][1] = {{0, -1, 0, 0, 0, 0}, {0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}};
  Fdag_blocks[1][2] = {{0}, {1}, {0}, {0}};
  Fdag_blocks[1][3] = {{0, 0, -1, 0}};
  Fdag_blocks[1][4] = {{0}};

  Fdag_blocks[2][0] = {{0, 0, 0, 0}, {-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};
  Fdag_blocks[2][1] = {{1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, -1, 0}};
  Fdag_blocks[2][2] = {{0}, {0}, {1}, {0}};
  Fdag_blocks[2][3] = {{0, 1, 0, 0}};
  Fdag_blocks[2][4] = {{0}};

  Fdag_blocks[3][0] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}};
  Fdag_blocks[3][1] = {{0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}};
  Fdag_blocks[3][2] = {{0}, {0}, {0}, {1}};
  Fdag_blocks[3][3] = {{-1, 0, 0, 0}};
  Fdag_blocks[3][4] = {{0}};

  for (int i = 0; i < 4; i++) {
    Fs.emplace_back(ann_conn, F_blocks[i]);
    Fdags.emplace_back(cre_conn, Fdag_blocks[i]);
  }

  // subspace indices
  std::vector<unsigned long> dummy2;
  std::vector<std::vector<unsigned long>> subspaces(num_blocks, dummy2);
  subspaces[0] = {1, 2, 4, 8};
  subspaces[1] = {3, 5, 6, 9, 10, 12};
  subspaces[2] = {0};
  subspaces[3] = {7, 11, 13, 14};
  subspaces[4] = {15};
  std::vector<long> fock_state_order(begin(subspaces[0]), end(subspaces[0]));
  for (int i = 1; i < num_blocks; i++) { fock_state_order.insert(end(fock_state_order), begin(subspaces[i]), end(subspaces[i])); }

  // Hamiltonian in dense storage
  auto H_dense    = nda::zeros<dcomplex>(16, 16);
  H_dense(0, 0)   = -1;
  H_dense(1, 1)   = -1;
  H_dense(2, 2)   = -1;
  H_dense(3, 3)   = -1;
  H_dense(4, 4)   = -0.6;
  H_dense(5, 8)   = 0.2;
  H_dense(6, 6)   = -0.4;
  H_dense(6, 7)   = 0.2;
  H_dense(7, 6)   = 0.2;
  H_dense(7, 7)   = -0.4;
  H_dense(8, 5)   = 0.2;
  H_dense(9, 9)   = -0.6;
  H_dense(11, 11) = 2;
  H_dense(12, 12) = 2;
  H_dense(13, 13) = 2;
  H_dense(14, 14) = 2;
  H_dense(15, 15) = 6;

  // Green's function in dense storage
  auto Gt_dense = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);

  // creation/annihilation operators in dense storage
  auto Fs_dense = nda::zeros<dcomplex>(4, 16, 16);
  // h5::read(hgroup, "c_dense", Fs_dense);
  // copied from a text dump of an h5 file output from atom_diag
  Fs_dense          = {{{0, 0, 0, 0, 0, -1, 0, 0, -2.23711e-17, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, -1, -2.23711e-17, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.23711e-17, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.23711e-17, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                       {{0, 0, 0, 0, 0, 0, -2.23711e-17, -1, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, -2.23711e-17, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.23711e-17, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.23711e-17, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                       {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 0, 0, 2.23711e-17, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 2.23711e-17, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.23711e-17, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.23711e-17, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                       {{0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 1, 2.23711e-17, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 2.23711e-17, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.23711e-17, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.23711e-17, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}};
  auto F_dags_dense = nda::zeros<dcomplex>(4, 16, 16);
  for (int i = 0; i < 4; i++) { F_dags_dense(i, _, _) = nda::transpose(nda::conj(Fs_dense(i, _, _))); }

  return std::make_tuple(num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order);
}

std::tuple<int, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, BlockDiagOpFun, std::vector<BlockOp>, std::vector<BlockOp>, nda::array<dcomplex, 3>,
           nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, std::vector<std::vector<unsigned long>>, std::vector<long>, BlockOpSymQuartet>
two_band_discrete_bath_helper_sym(double beta, double Lambda, double eps) {

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  int n = 4, k = num_blocks;
  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // create cre/ann operators
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);

  // TODO replace this with read from atom_diag
  // compute creation and annihilation operators in block-sparse storage
  auto F_sym = BlockOpSymSet(n, Fs[0].get_block_indices(), Fs[0].get_block_sizes());
  for (int i = 0; i < k; i++) {
    if (Fs[0].get_block_index(i) == -1) continue; // skip empty blocks
    auto Fs_sym_block = nda::zeros<dcomplex>(n, Fs[0].get_block_size(i, 0), Fs[0].get_block_size(i, 1));
    for (int j = 0; j < n; j++) { Fs_sym_block(j, _, _) = Fs[j].get_block(i); }
    F_sym.set_block(i, Fs_sym_block);
  }
  std::vector<BlockOpSymSet> F_sym_vec{F_sym};

  auto F_dag_sym = BlockOpSymSet(n, Fdags[0].get_block_indices(), Fdags[0].get_block_sizes());
  for (int i = 0; i < k; i++) {
    if (Fdags[0].get_block_index(i) == -1) continue; // skip empty blocks
    auto F_dags_sym_block = nda::zeros<dcomplex>(n, Fdags[0].get_block_size(i, 0), Fdags[0].get_block_size(i, 1));
    for (int j = 0; j < n; j++) { F_dags_sym_block(j, _, _) = Fdags[j].get_block(i); }
    F_dag_sym.set_block(i, F_dags_sym_block);
  }
  std::vector<BlockOpSymSet> F_dag_sym_vec{F_dag_sym};
  nda::vector<long> sym_set_labels_triv(n);
  sym_set_labels_triv = 0;

  auto Fq = BlockOpSymQuartet(F_sym_vec, F_dag_sym_vec, hyb_coeffs, hyb_refl_coeffs, sym_set_labels_triv);

  return std::make_tuple(num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order, Fq);
}

std::tuple<nda::array<dcomplex, 3>, DenseFSet, std::vector<nda::vector<int>>>
spin_flip_fermion_dense_helper(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                               nda::array_const_view<dcomplex, 3> hyb_refl_coeffs, std::string filename) {
  // DLR generation
  auto dlr_rf     = build_dlr_rf(Lambda, eps);
  auto itops      = imtime_ops(Lambda, dlr_rf);
  auto dlr_it     = itops.get_itnodes();
  auto dlr_it_abs = rel2abs(dlr_it);

  h5::file f(filename, 'r');
  h5::group g(f);

  long n = 0;
  h5::read(g, "norb", n);
  int N                           = static_cast<int>(pow(4, static_cast<double>(n))); // number of Fock states
  nda::array<dcomplex, 2> H_dense = nda::zeros<dcomplex>(N, N);
  h5::read(g, "H_mat_dense", H_dense);
  auto Gt_dense                    = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  nda::array<dcomplex, 3> Fs_dense = nda::zeros<dcomplex>(2 * n, N, N);
  h5::read(g, "c_dense", Fs_dense);
  nda::array<dcomplex, 3> F_dags_dense = nda::zeros<dcomplex>(2 * n, N, N);
  h5::read(g, "cdag_dense", F_dags_dense);

  DenseFSet Fset(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  long k = 0;
  h5::read(g, "num_blocks", k);
  std::vector<nda::vector<int>> subspaces(k, nda::vector<int>());
  for (int i = 0; i < k; ++i) {
    nda::vector<int> subspace;
    h5::read(g, "ad/sub_hilbert_spaces/" + std::to_string(i) + "/fock_states", subspace);
    subspaces[i] = subspace;
  }

  return std::make_tuple(Gt_dense, Fset, subspaces);
}

TEST(BlockSparseNCA, simple) {
  // set up arguments to block_sparse/NCA_bs()
  int N = 4;
  int r = 1;
  int n = 2;

  // set up hybridization
  nda::array<dcomplex, 3> hyb({r, n, n});
  nda::array<dcomplex, 3> hyb_refl({r, n, n});
  for (int t = 0; t < r; ++t) {
    hyb(t, 0, 0)      = 1;
    hyb(t, 1, 1)      = -1;
    hyb(t, 0, 1)      = -1;
    hyb(t, 1, 0)      = 4;
    hyb_refl(t, _, _) = nda::transpose(hyb(t, _, _));
  }

  // set up Green's function
  dcomplex mu = 0.2789;
  dcomplex U  = 1.01;
  dcomplex V  = 0.123;
  nda::array<dcomplex, 3> block0({r, 1, 1});
  nda::array<dcomplex, 3> block1({r, 2, 2});
  nda::array<dcomplex, 3> block2({r, 1, 1});
  for (int t = 0; t < r; ++t) {
    block0(t, 0, 0) = 0;
    block1(t, 0, 0) = mu;
    block1(t, 1, 1) = mu;
    block1(t, 0, 1) = V;
    block1(t, 1, 0) = V;
    block2(t, 0, 0) = 2 * mu + U;
  }
  std::vector<nda::array<dcomplex, 3>> Gt_blocks = {block0, block1, block2};
  nda::vector<int> zero_block_indices            = {-1, 0, 0};
  BlockDiagOpFun Gt(Gt_blocks, zero_block_indices);

  // set up annihilation operators
  nda::vector<int> block_indices_F = {-1, 0, 1};

  nda::array<dcomplex, 2> F_up_block0              = {{0}};
  nda::array<dcomplex, 2> F_up_block1              = {{1, 0}};
  nda::array<dcomplex, 2> F_up_block2              = {{0}, {1}};
  std::vector<nda::array<dcomplex, 2>> F_up_blocks = {F_up_block0, F_up_block1, F_up_block2};
  BlockOp F_up(block_indices_F, F_up_blocks);

  nda::array<dcomplex, 2> F_down_block0              = {{0}};
  nda::array<dcomplex, 2> F_down_block1              = {{0, 1}};
  nda::array<dcomplex, 2> F_down_block2              = {{-1}, {0}};
  std::vector<nda::array<dcomplex, 2>> F_down_blocks = {F_down_block0, F_down_block1, F_down_block2};
  BlockOp F_down(block_indices_F, F_down_blocks);

  std::vector<BlockOp> Fs   = {F_up, F_down};
  BlockDiagOpFun NCA_result = NCA_bs(hyb, hyb_refl, Gt, Fs);

  // compute NCA_result using dense storage

  nda::array<dcomplex, 3> Gt_dense({r, N, N});
  Gt_dense(0, 0, 0) = 0;
  Gt_dense(0, 1, 1) = mu;
  Gt_dense(0, 2, 2) = mu;
  Gt_dense(0, 1, 2) = V;
  Gt_dense(0, 2, 1) = V;
  Gt_dense(0, 3, 3) = 2 * mu + U;

  nda::array<dcomplex, 2> F_up_dense({N, N});
  F_up_dense(0, 1) = 1;
  F_up_dense(2, 3) = 1;

  nda::array<dcomplex, 2> F_down_dense({N, N});
  F_down_dense(0, 2) = 1;
  F_down_dense(1, 3) = -1;

  nda::array<dcomplex, 2> F_up_dag_dense   = nda::transpose(F_up_dense);
  nda::array<dcomplex, 2> F_down_dag_dense = nda::transpose(F_down_dense);

  auto NCA_result_dense = nda::zeros<dcomplex>(r, N, N);
  nda::array<dcomplex, 2> temp_dense({N, N});
  for (int t = 0; t < r; ++t) {
    // backward diagram
    temp_dense = nda::matmul(F_up_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 0) * nda::matmul(temp_dense, F_up_dag_dense);
    temp_dense = nda::matmul(F_up_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 1) * nda::matmul(temp_dense, F_down_dag_dense);
    temp_dense = nda::matmul(F_down_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 0) * nda::matmul(temp_dense, F_up_dag_dense);
    temp_dense = nda::matmul(F_down_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 1) * nda::matmul(temp_dense, F_down_dag_dense);

    // forward diagram
    temp_dense = nda::matmul(F_up_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 0) * nda::matmul(temp_dense, F_up_dense);
    temp_dense = nda::matmul(F_up_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 1) * nda::matmul(temp_dense, F_down_dense);
    temp_dense = nda::matmul(F_down_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 0) * nda::matmul(temp_dense, F_up_dense);
    temp_dense = nda::matmul(F_down_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 1) * nda::matmul(temp_dense, F_down_dense);
  }

  EXPECT_EQ(NCA_result.get_block(0)(_, 0, 0), NCA_result_dense(_, 0, 0));
  EXPECT_EQ(NCA_result.get_block(1), NCA_result_dense(_, range(1, 3), range(1, 3)));
  EXPECT_EQ(NCA_result.get_block(2)(_, 0, 0), NCA_result_dense(_, 3, 3));
}

TEST(BlockSparseNCA, single_exponential) {
  // DLR parameters
  double beta        = 1.0;
  double Lambda      = 100.0;
  double eps         = 1.0e-13;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();

  auto dlr_it_abs = cppdlr::rel2abs(dlr_it);

  // create hybridization
  double D             = 0.03;
  auto Deltat          = nda::array<dcomplex, 3>(r, 1, 1);
  auto Deltat_refl     = nda::array<dcomplex, 3>(r, 1, 1);
  Deltat(_, 0, 0)      = exp(-D * dlr_it_abs * beta);
  Deltat_refl(_, 0, 0) = exp(D * dlr_it_abs * beta);

  // create Green's function
  double g                                       = -0.54;
  auto Gt_block                                  = nda::array<dcomplex, 3>(r, 1, 1);
  auto Gt_zero_block_index                       = nda::ones<int>(1);
  Gt_block(_, 0, 0)                              = exp(-g * dlr_it_abs * beta);
  std::vector<nda::array<dcomplex, 3>> Gt_blocks = {Gt_block};
  auto Gt                                        = BlockDiagOpFun(Gt_blocks, Gt_zero_block_index);

  // create annihilation operator
  auto F_block                                  = nda::ones<dcomplex>(1, 1);
  auto F_block_indices                          = nda::vector<int>(1);
  F_block_indices                               = 0;
  std::vector<nda::array<dcomplex, 2>> F_blocks = {F_block};
  auto F                                        = BlockOp(F_block_indices, F_blocks);
  std::vector<BlockOp> Fs                       = {F};

  BlockDiagOpFun NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);
  auto NCA_ana              = nda::zeros<dcomplex>(r);
  for (int i = 0; i < r; i++) {
    auto tau   = dlr_it_abs(i);
    NCA_ana(i) = -exp(-(D + g) * tau) - exp((D - g) * tau);
  }

  EXPECT_LT(nda::norm((NCA_result.get_block(0)(_, 0, 0) - NCA_ana), std::numeric_limits<double>::infinity())
               / nda::norm(NCA_ana, std::numeric_limits<double>::infinity()),
            1.0e-12);
}

TEST(BlockSparseNCA, two_band_discrete_bath_bs_vs_dense) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // block-sparse NCA compuation
  auto NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);

  // dense-matrix NCA computation
  auto NCA_dense_result = NCA_dense(Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense);

  // check that block-sparse NCA calculation agrees with dense NCA calculation
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(i) - NCA_dense_result(_, range(s0, s1), range(s0, s1)))), 10 * eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseNCA, PYTHON_two_band_discrete_bath_bs_vs_py) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // block-sparse NCA compuation
  auto NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);

  // load NCA twoband.py
  h5::file Gtfile("../test/c++/h5/two_band_py.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) { NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]); }
    }
  }

  // check that NCA calculation agrees with python NCA calculation
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(i) - NCA_py_perm(_, range(s0, s1), range(s0, s1)))), 10 * eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseOCA, single_exponential) {
  // DLR parameters
  double beta        = 1.0;
  double Lambda      = 100.0;
  double eps         = 1.0e-13;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();

  auto dlr_it_abs = cppdlr::rel2abs(dlr_it);

  // create hybridization
  double D    = -10.0;
  auto Deltat = nda::array<dcomplex, 3>(r, 1, 1);
  // Deltat(_,0,0) = exp(-D*dlr_it_abs*beta);
  for (int t = 0; t < r; t++) Deltat(t, 0, 0) = k_it(dlr_it(t), D);

  // create Green's function
  double g                 = -13.0;
  auto Gt_block            = nda::array<dcomplex, 3>(r, 1, 1);
  auto Gt_zero_block_index = nda::ones<int>(1);
  for (int t = 0; t < r; t++) Gt_block(t, 0, 0) = k_it(dlr_it(t), g);
  std::vector<nda::array<dcomplex, 3>> Gt_blocks = {Gt_block};
  auto Gt                                        = BlockDiagOpFun(Gt_blocks, Gt_zero_block_index);

  // create annihilation operator
  auto F_block                                  = nda::ones<dcomplex>(1, 1);
  auto F_block_indices                          = nda::vector<int>(1);
  F_block_indices                               = 0;
  std::vector<nda::array<dcomplex, 2>> F_blocks = {F_block};
  auto F                                        = BlockOp(F_block_indices, F_blocks);
  std::vector<BlockOp> Fs                       = {F};

  auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);
  auto OCA_ana    = nda::zeros<dcomplex>(r);
  for (int i = 0; i < r; i++) {
    auto tau = dlr_it_abs(i);
    // ff term
    OCA_ana(i) = exp(-g * (-3 + tau) - 2 * D * (-1 + tau)) * (1 + exp(D * tau) * (-1 + D * tau))
       / (D * D * (1 + exp(D)) * (1 + exp(D)) * (1 + exp(g)) * (1 + exp(g)) * (1 + exp(g)));
    // fb term
    OCA_ana(i) += exp(D + 3 * g - (D + g) * tau) * (-1 + exp(D * tau)) * (-1 + exp(D * tau))
       / (2 * D * D * (1 + exp(D)) * (1 + exp(D)) * (1 + exp(g)) * (1 + exp(g)) * (1 + exp(g)));
    // bf term
    OCA_ana(i) += exp(D + 3 * g - (D + g) * tau) * (-1 + exp(D * tau)) * (-1 + exp(D * tau))
       / (2 * D * D * (1 + exp(D)) * (1 + exp(D)) * (1 + exp(g)) * (1 + exp(g)) * (1 + exp(g)));
    // bb term
    OCA_ana(i) += -exp(-g * (-3 + tau) + D * tau) * (1 - exp(D * tau) + D * tau)
       / (D * D * (1 + exp(D)) * (1 + exp(D)) * (1 + exp(g)) * (1 + exp(g)) * (1 + exp(g)));
  }
  EXPECT_LT(nda::norm((OCA_result.get_block(0)(_, 0, 0) - OCA_ana), std::numeric_limits<double>::infinity()), 1.0e-7);
}

TEST(BlockSparseMisc, compute_nonint_gf) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 1000 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // the following variables can be read from the output of benchmarks/atom_diag_to_text.py
  int num_blocks = 5;                                      // number of blocks in Hamiltonian
  std::vector<nda::array<double, 2>> H_blocks(num_blocks); // Hamiltonian in sparse storage
  H_blocks[0]                           = nda::make_regular(-1 * nda::eye<double>(4));
  H_blocks[1]                           = {{-0.6, 0, 0, 0, 0, 0},   {0, 8.27955e-19, 0, 0, 0.2, 0}, {0, 0, -0.4, 0.2, 0, 0},
                                           {0, 0, 0.2, -0.4, 0, 0}, {0, 0.2, 0, 0, 8.27955e-19, 0}, {0, 0, 0, 0, 0, -0.6}};
  H_blocks[2]                           = {{0}};
  H_blocks[3]                           = nda::make_regular(2 * nda::eye<double>(4));
  H_blocks[4]                           = {{6}};
  nda::vector<int> H_block_inds         = {0, 0, -1, 0, 0};
  auto H_dense                          = nda::zeros<dcomplex>(16, 16); // Hamiltonian in dense storage
  H_dense(range(0, 4), range(0, 4))     = H_blocks[0];
  H_dense(range(4, 10), range(4, 10))   = H_blocks[1];
  H_dense(range(11, 15), range(11, 15)) = H_blocks[3];
  H_dense(15, 15)                       = 6;

  // compute noninteracting Green's function from dense Hamiltonian
  auto [H_loc_eval, H_loc_evec] = nda::linalg::eigenelements(H_dense);
  auto E0                       = nda::min_element(H_loc_eval);
  H_loc_eval -= E0;
  auto tr_exp_minusbetaH = nda::sum(exp(-beta * H_loc_eval));
  auto eta_0             = nda::log(tr_exp_minusbetaH) / beta;
  H_loc_eval += eta_0;
  auto Gt_evals_t = nda::zeros<dcomplex>(16, 16);
  auto Gt_mat     = nda::zeros<dcomplex>(r, 16, 16);
  auto Gbeta      = nda::zeros<dcomplex>(16, 16);
  Gt_mat          = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  for (int i = 0; i < 16; i++) { Gbeta(i, i) = -exp(-beta * H_loc_eval(i)); }
  Gbeta = nda::matmul(Gbeta, nda::transpose(H_loc_evec));
  Gbeta = nda::matmul(H_loc_evec, Gbeta);
  // check that trace of noninteracting Green's function from dense
  // Hamiltonian at tau = beta has trace 1
  ASSERT_LE(nda::abs(nda::trace(Gbeta) + 1), 1e-13);

  auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);
  // check that the noninteracting Green's function, computing from the
  // sparse- and dense-storage Hamiltonians are the same
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(0, 4), range(0, 4)) - Gt.get_block(0))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(4, 10), range(4, 10)) - Gt.get_block(1))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(10, 11), range(10, 11)) - Gt.get_block(2))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(11, 15), range(11, 15)) - Gt.get_block(3))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(15, 16), range(15, 16)) - Gt.get_block(4))), 1e-13);
}

TEST(BlockSparseMisc, PYTHON_compute_nonint_gf) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 1000 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // the following variables can be read from the output of benchmarks/atom_diag_to_text.py
  int num_blocks = 5;                                      // number of blocks in Hamiltonian
  std::vector<nda::array<double, 2>> H_blocks(num_blocks); // Hamiltonian in sparse storage
  H_blocks[0]                           = nda::make_regular(-1 * nda::eye<double>(4));
  H_blocks[1]                           = {{-0.6, 0, 0, 0, 0, 0},   {0, 8.27955e-19, 0, 0, 0.2, 0}, {0, 0, -0.4, 0.2, 0, 0},
                                           {0, 0, 0.2, -0.4, 0, 0}, {0, 0.2, 0, 0, 8.27955e-19, 0}, {0, 0, 0, 0, 0, -0.6}};
  H_blocks[2]                           = {{0}};
  H_blocks[3]                           = nda::make_regular(2 * nda::eye<double>(4));
  H_blocks[4]                           = {{6}};
  nda::vector<int> H_block_inds         = {0, 0, -1, 0, 0};
  auto H_dense                          = nda::zeros<dcomplex>(16, 16); // Hamiltonian in dense storage
  H_dense(range(0, 4), range(0, 4))     = H_blocks[0];
  H_dense(range(4, 10), range(4, 10))   = H_blocks[1];
  H_dense(range(11, 15), range(11, 15)) = H_blocks[3];
  H_dense(15, 15)                       = 6;

  // load noninteracting Green's function from hdf5 file, produced from a
  // run of benchmarks/twoband.py
  h5::file hfile2("../test/c++/h5/two_band_py.h5", 'r');
  h5::group hgroup2(hfile2);
  auto G0_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(hgroup2, "G0_iaa", G0_py);

  // compute noninteracting Green's function from dense Hamiltonian
  auto [H_loc_eval, H_loc_evec] = nda::linalg::eigenelements(H_dense);
  auto E0                       = nda::min_element(H_loc_eval);
  H_loc_eval -= E0;
  auto tr_exp_minusbetaH = nda::sum(exp(-beta * H_loc_eval));
  auto eta_0             = nda::log(tr_exp_minusbetaH) / beta;
  H_loc_eval += eta_0;
  auto Gt_evals_t = nda::zeros<dcomplex>(16, 16);
  auto Gt_mat     = nda::zeros<dcomplex>(r, 16, 16);
  auto Gbeta      = nda::zeros<dcomplex>(16, 16);
  Gt_mat          = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  for (int i = 0; i < 16; i++) { Gbeta(i, i) = -exp(-beta * H_loc_eval(i)); }
  Gbeta = nda::matmul(Gbeta, nda::transpose(H_loc_evec));
  Gbeta = nda::matmul(H_loc_evec, Gbeta);
  // check that trace of noninteracting Green's function from dense
  // Hamiltonian at tau = beta has trace 1
  ASSERT_LE(nda::abs(nda::trace(Gbeta) + 1), 1e-13);

  auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);
  // check that the noninteracting Green's function, computing from the
  // sparse- and dense-storage Hamiltonians are the same
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(0, 4), range(0, 4)) - Gt.get_block(0))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(4, 10), range(4, 10)) - Gt.get_block(1))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(10, 11), range(10, 11)) - Gt.get_block(2))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(11, 15), range(11, 15)) - Gt.get_block(3))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(15, 16), range(15, 16)) - Gt.get_block(4))), 1e-13);

  // check that the result here agrees with the result of benchmarks/twoband.py
  ASSERT_LE(nda::max_element(nda::abs(G0_py - Gt_mat)), 1e-13);
}

TEST(BlockSparseOCA, PYTHON_two_band_discrete_bath_bs) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // block-sparse NCA and OCA computations
  auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);
  // load NCA and OCA results from twoband.py
  h5::file Gtfile("../test/c++/h5/two_band_py.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);
  auto OCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "OCA", OCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j) = OCA_py(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }

  // check that block-sparse OCA calculation agrees with twoband.py
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(
                 nda::abs(OCA_result.get_block(i) - OCA_py_perm(_, range(s0, s1), range(s0, s1)) + NCA_py_perm(_, range(s0, s1), range(s0, s1)))),
              eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseOCA, PYTHON_two_band_discrete_bath_dense) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // dense-matrix OCA computation
  auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // load NCA and OCA results from twoband.py
  h5::file Gtfile("../test/c++/h5/two_band_py.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);
  auto OCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "OCA", OCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j) = OCA_py(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }

  // check that dense OCA calculation agree with twoband.py
  ASSERT_LE(nda::max_element(nda::abs(OCA_dense_result - OCA_py_perm + NCA_py_perm)), eps);
}

TEST(BlockSparseOCA, two_band_discrete_bath_bs_vs_dense) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // block-sparse OCA computation
  auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);

  // dense-matrix OCA computation
  auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // check that block-sparse OCA calculation agrees with dense OCA calculation
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - OCA_dense_result(_, range(s0, s1), range(s0, s1)))), eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseOCA, PYTHON_two_band_discrete_bath_tpz) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 1000 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // block-sparse OCA compuation
  auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);

  int n_quad = 100;
  // compute OCA using trapezoidal rule using 100 quadrature nodes0
  // load precomputed values from the following 3 lines:
  // auto OCA_tpz_result = OCA_tpz(Deltat, itops, beta, Gt_dense, Fs_dense, n_quad);
  // h5::file tpz_file("../test/c++/h5/tpz100.h5", 'w');
  // h5::write(tpz_file, "OCA_tpz_result", OCA_tpz_result);
  h5::file tpz_file("../test/c++/h5/tpz100.h5", 'r');
  nda::array<dcomplex, 3> OCA_tpz_result(101, 16, 16);
  h5::read(tpz_file, "OCA_tpz_result", OCA_tpz_result);

  // check that trapezoidal OCA calculation agrees with block-sparse calc.
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    auto OCA_result_block    = OCA_result.get_block(i)(_, _, _);
    auto OCA_result_block_eq = eval_eq(itops, OCA_result_block, n_quad);
    ASSERT_LE(nda::max_element(nda::abs(OCA_result_block_eq - OCA_tpz_result(_, range(s0, s1), range(s0, s1)))), 2e-4);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(DenseBackbone, one_vertex_and_edge) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-6;
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // create cre/ann operators
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  auto D = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, Gt_dense, Fset);
  for (int fb1 = 0; fb1 <= 1; fb1++) {
    // initialize backbone
    auto B = Backbone(topology, n);

    // set line directions
    nda::vector<int> fb = {1, fb1, 0};
    B.set_directions(fb);

    // set pole indices
    nda::vector<int> pole_inds = {0, r - 1};
    B.set_pole_inds(pole_inds, dlr_rf);

    // set orbital indices
    nda::vector<int> orb_inds = {1, 0, 1, 2, 0, 2};
    B.set_orb_inds(orb_inds);

    // multiply T by vertex 1
    for (int t = 0; t < r; t++) D.T(t, _, _) = nda::eye<dcomplex>(N);
    D.multiply_vertex_dense(B, 1);
    std::cout << B << std::endl;

    // do the same multiplication manually
    nda::array<dcomplex, 3> Tact(r, N, N);
    if (fb1 == 1) {
      for (int t = 0; t < r; t++) Tact(t, _, _) = k_it(dlr_it(t), -dlr_rf(pole_inds(0))) * Fs_dense(0, _, _);
    } else {
      for (int t = 0; t < r; t++) Tact(t, _, _) = F_dags_dense(0, _, _);
    }
    ASSERT_LE(nda::max_element(nda::abs(D.T - Tact)), 1e-12);

    // check that convolution with function on first edge is correct
    D.compose_with_edge_dense(B, 1);
    if (fb1 == 1) {
      Tact = itops.convolve(beta, Fermion, itops.vals2coefs(Gt_dense), itops.vals2coefs(Tact), TIME_ORDERED);
    } else {
      nda::array<dcomplex, 3> GKt_act(r, N, N);
      for (int t = 0; t < r; t++) GKt_act(t, _, _) = k_it(dlr_it(t), -dlr_rf(pole_inds(0))) * Gt_dense(t, _, _);
      Tact = itops.convolve(beta, Fermion, itops.vals2coefs(GKt_act), itops.vals2coefs(Tact), TIME_ORDERED);
    }
    ASSERT_LE(nda::max_element(nda::abs(D.T - Tact)), 1e-12);
  }
}

TEST(DenseBackbone, OCA) {
  int n         = 4;
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;

  // load in functions from two_band.py
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  // compute Fbars and Fdagbars and store in Fset
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // initialize Backbone and DiagramEvaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  auto D                      = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, Gt_dense, Fset);

  // evaluate OCA self-energy contribution
  D.eval_diagram_dense(B);
  auto OCA_result = D.Sigma;

  // compare against manually-computed OCA result
  auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_dense_result)), eps);
}

TEST(DenseBackbone, third_order_manual) {
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 10.0 * beta; // 1000.0*beta;
  double eps    = 1.0e-10;
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // compute hybridization function
  auto hyb             = Deltat;
  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(hyb));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // compute self-energy contribution of one third-order diagram topology,
  // with all forward hybridization lines and particular poles
  auto Sigma_manual           = third_order_dense_partial(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  auto B                      = Backbone(topology, n);
  nda::vector<int> fb{1, 1, 1}, pole_inds{7, 9};
  B.set_directions(fb);
  B.set_pole_inds(pole_inds, dlr_rf);
  auto D = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, Gt_dense, Fset);

  // perform the same calculation using the a routine called by eval_diagram_dense()
  nda::array<dcomplex, 3> T(r, N, N), GKt(r, N, N), Tmu(r, N, N), Sigma_generic(r, N, N);
  nda::array<dcomplex, 4> Tkaps(n, r, N, N);
  nda::vector<int> states(6);
  // f_ix = o_ix + n^(m-1) * p_ix + (n * r)^(m-1) * fb_ix
  int pow_n_mm1  = static_cast<int>(std::pow(n, B.m - 1));
  int pow_nr_mm1 = static_cast<int>(std::pow(n * r, B.m - 1));
  int f_ix_start = pow_n_mm1 * (pole_inds(0) + r * pole_inds(1)) + pow_nr_mm1 * (fb(0) + 2 * fb(1) + 4 * fb(2));
  for (int f_ix_off = 0; f_ix_off < pow_n_mm1; f_ix_off++) {
    B.set_flat_index(f_ix_start + f_ix_off, dlr_rf);
    D.eval_backbone_fixed_indices_dense(B);
    B.reset_all_inds();
  }

  ASSERT_LE(nda::max_element(nda::abs(Sigma_manual(10, _, _) - D.Sigma(10, _, _))), eps);
}

TEST(DenseBackbone, PYTHON_third_order) {
  nda::array<int, 3> topologies = {{{0, 2}, {1, 4}, {3, 5}}, {{0, 3}, {1, 5}, {2, 4}}, {{0, 4}, {1, 3}, {2, 5}}, {{0, 3}, {1, 4}, {2, 5}}};
  nda::vector<int> topo_sign{1, 1, 1, -1}; // topo_sign(i) = (-1)^{# of line crossings in topology i}

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 10.0 * beta; // 1000.0*beta;
  double eps    = 1.0e-10;
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // compute Fbars and Fdagbars
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // compute NCA and OCA
  auto NCA_result          = NCA_dense(Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense);
  nda::array<int, 2> T_OCA = {{0, 2}, {1, 3}};
  auto B_OCA               = Backbone(T_OCA, n);
  auto D                   = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, Gt_dense, Fset); // create DiagramEvaluator object
  D.eval_diagram_dense(B_OCA);                                                                   // evaluate OCA diagram
  auto OCA_result = D.Sigma;                                                                     // get the result from the DiagramEvaluator
  D.reset();

  // arrays for storing results from third-order diagram computations
  auto third_order_result      = nda::zeros<dcomplex>(r, N, N);
  auto third_order_02_result   = nda::zeros<dcomplex>(r, N, N);
  auto third_order_0314_result = nda::zeros<dcomplex>(r, N, N);
  auto third_order_0315_result = nda::zeros<dcomplex>(r, N, N);
  auto third_order_04_result   = nda::zeros<dcomplex>(r, N, N);

  // compute third-order diagrams using generic backbone evaluators
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 4; i++) {
    auto B = Backbone(topologies(i, _, _), n); // create Backbone object for topology i
    D.eval_diagram_dense(B);
    auto eval = D.Sigma;
    third_order_result += topo_sign(i) * D.Sigma;
    if (i == 0)
      third_order_02_result = eval;
    else if (i == 1)
      third_order_0315_result = eval;
    else if (i == 2)
      third_order_04_result = eval;
    else
      third_order_0314_result = eval;
    D.reset();
  }
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "Elapsed time for dense comp'n of 3rd order diags = " << duration.count() << " seconds" << std::endl;

  // load results from a run of twoband.py
  h5::file hfile("../test/c++/h5/two_band_py_Lambda10.h5", 'r');
  h5::group hgroup(hfile);
  nda::array<dcomplex, 3> NCA_py(r, N, N), OCA_py(r, N, N);
  nda::array<dcomplex, 3> third_order_py(r, N, N);
  nda::array<dcomplex, 3> third_order_py_02(r, N, N);
  nda::array<dcomplex, 3> third_order_py_0314(r, N, N);
  nda::array<dcomplex, 3> third_order_py_0315(r, N, N);
  nda::array<dcomplex, 3> third_order_py_04(r, N, N);
  h5::read(hgroup, "NCA", NCA_py);
  h5::read(hgroup, "OCA", OCA_py);
  OCA_py = OCA_py - NCA_py;

  h5::read(hgroup, "third_order", third_order_py);
  third_order_py = -third_order_py + OCA_py + NCA_py;
  h5::read(hgroup, "third_order_[(0, 2), (1, 4), (3, 5)]", third_order_py_02);
  h5::read(hgroup, "third_order_[(0, 3), (1, 4), (2, 5)]", third_order_py_0314);
  h5::read(hgroup, "third_order_[(0, 3), (1, 5), (2, 4)]", third_order_py_0315);
  h5::read(hgroup, "third_order_[(0, 4), (1, 3), (2, 5)]", third_order_py_04);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm              = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm              = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_perm      = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_02_perm   = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_0314_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_0315_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_04_perm   = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j)              = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j)              = OCA_py(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_perm(t, i, j)      = third_order_py(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_02_perm(t, i, j)   = third_order_py_02(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_0314_perm(t, i, j) = third_order_py_0314(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_0315_perm(t, i, j) = third_order_py_0315(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_04_perm(t, i, j)   = third_order_py_04(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }

  ASSERT_LE(nda::max_element(nda::abs(NCA_result - NCA_py_perm)), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_py_perm)), eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_02_result - third_order_py_02_perm)), 100 * eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_0314_result - third_order_py_0314_perm)), 100 * eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_0315_result - third_order_py_0315_perm)), 100 * eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_04_result - third_order_py_04_perm)), 100 * eps);

  ASSERT_LE(nda::max_element(nda::abs(third_order_result - third_order_py_perm)), 100 * eps);
}

TEST(Backbone, flat_index) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  int n                       = 4;

  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  int r       = dlr_rf.size();

  nda::vector<int> fb{1, 1, 0};
  nda::vector<int> pole_inds{3, 10};
  nda::vector<int> orb_inds{1, 3, 1, 2, 3, 2};

  auto B = Backbone(topology, n);
  B.set_directions(fb);
  B.set_pole_inds(pole_inds, dlr_rf);
  B.set_orb_inds(orb_inds);

  int fb_ix = 1 + 2 * 1;
  int p_ix  = 3 + r * 10;
  int o_ix  = 3 + n * 2;
  auto B2   = Backbone(topology, n);
  B2.set_directions(fb_ix);
  B2.set_pole_inds(p_ix, dlr_rf);
  B2.set_orb_inds(o_ix);

  ASSERT_EQ(B.get_fb(0), B2.get_fb(0));
  ASSERT_EQ(B.get_fb(1), B2.get_fb(1));
  ASSERT_EQ(B.get_fb(2), B2.get_fb(2));
  ASSERT_EQ(B.get_pole_ind(0), B2.get_pole_ind(0));
  ASSERT_EQ(B.get_pole_ind(1), B2.get_pole_ind(1));
  ASSERT_EQ(B.get_orb_ind(1), B2.get_orb_ind(1));
  ASSERT_EQ(B.get_orb_ind(3), B2.get_orb_ind(3));
  ASSERT_EQ(B.get_orb_ind(4), B2.get_orb_ind(4));
  ASSERT_EQ(B.get_orb_ind(5), B2.get_orb_ind(5));

  int f_ix = o_ix + p_ix * n * n + fb_ix * n * n * r * r;
  auto B3  = Backbone(topology, n);
  B3.set_flat_index(f_ix, dlr_rf);

  ASSERT_EQ(B.get_fb(0), B3.get_fb(0));
  ASSERT_EQ(B.get_fb(1), B3.get_fb(1));
  ASSERT_EQ(B.get_fb(2), B3.get_fb(2));
  ASSERT_EQ(B.get_pole_ind(0), B3.get_pole_ind(0));
  ASSERT_EQ(B.get_pole_ind(1), B3.get_pole_ind(1));
  ASSERT_EQ(B.get_orb_ind(1), B3.get_orb_ind(1));
  ASSERT_EQ(B.get_orb_ind(3), B3.get_orb_ind(3));
  ASSERT_EQ(B.get_orb_ind(4), B3.get_orb_ind(4));
  ASSERT_EQ(B.get_orb_ind(5), B3.get_orb_ind(5));
}

TEST(Backbone, data_structures) {
  int n = 4, k = 5;
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // compute hybridization function
  auto hyb             = Deltat;
  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(hyb));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);

  // compute creation and annihilation operators in dense storage
  auto Fset = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // compute creation and annihilation operators in block-sparse storage
  auto F_sym = BlockOpSymSet(n, Fs[0].get_block_indices(), Fs[0].get_block_sizes());
  for (int i = 0; i < k; i++) {
    if (Fs[0].get_block_index(i) == -1) continue; // skip empty blocks
    auto Fs_sym_block = nda::zeros<dcomplex>(n, Fs[0].get_block_size(i, 0), Fs[0].get_block_size(i, 1));
    for (int j = 0; j < n; j++) { Fs_sym_block(j, _, _) = Fs[j].get_block(i); }
    F_sym.set_block(i, Fs_sym_block);
  }
  std::vector<BlockOpSymSet> F_sym_vec{F_sym};

  auto F_dag_sym = BlockOpSymSet(n, Fdags[0].get_block_indices(), Fdags[0].get_block_sizes());
  for (int i = 0; i < k; i++) {
    if (Fdags[0].get_block_index(i) == -1) continue; // skip empty blocks
    auto F_dags_sym_block = nda::zeros<dcomplex>(n, Fdags[0].get_block_size(i, 0), Fdags[0].get_block_size(i, 1));
    for (int j = 0; j < n; j++) { F_dags_sym_block(j, _, _) = Fdags[j].get_block(i); }
    F_dag_sym.set_block(i, F_dags_sym_block);
  }
  std::vector<BlockOpSymSet> F_dag_sym_vec{F_dag_sym};

  // TODO test compares Fs_sym and F_dags_sym here with read from atom_diag

  // compute F^{dag bar} and F^bar in block-sparse storage
  auto F_dag_bar_sym = BlockOpSymSetBar(n, r, F_dag_sym.get_block_indices(), F_dag_sym.get_block_sizes());
  for (int i = 0; i < k; i++) {
    if (F_dag_sym.get_block_index(i) == -1) continue; // skip empty blocks
    auto F_dags_bar_sym_block = nda::zeros<dcomplex>(n, r, F_dag_sym.get_block_size(i, 0), F_dag_sym.get_block_size(i, 1));
    for (int nu = 0; nu < n; nu++) {
      for (int lam = 0; lam < n; lam++) {
        for (int l = 0; l < r; l++) { F_dags_bar_sym_block(lam, l, _, _) += hyb_coeffs(l, nu, lam) * F_dag_sym.get_block(i)(nu, _, _); }
      }
    }
    F_dag_bar_sym.set_block(i, F_dags_bar_sym_block);
  }

  // compute F^{bar} in block-sparse storage
  auto F_bar_sym = BlockOpSymSetBar(n, r, F_sym.get_block_indices(), F_sym.get_block_sizes());
  for (int i = 0; i < k; i++) {
    if (F_sym.get_block_index(i) == -1) continue; // skip empty blocks
    auto F_bar_sym_block = nda::zeros<dcomplex>(n, r, F_sym.get_block_size(i, 0), F_sym.get_block_size(i, 1));
    for (int lam = 0; lam < n; lam++) {
      for (int nu = 0; nu < n; nu++) {
        for (int l = 0; l < r; l++) { F_bar_sym_block(nu, l, _, _) += hyb_refl_coeffs(l, lam, nu) * F_sym.get_block(i)(lam, _, _); }
      }
    }
    F_bar_sym.set_block(i, F_bar_sym_block);
  }

  std::cout << F_dag_bar_sym.get_block(0)(0, 10, _, _) << std::endl;
  auto sym_set_labels = nda::zeros<long>(n);
  auto F_quartet      = BlockOpSymQuartet(F_sym_vec, F_dag_sym_vec, hyb_coeffs, hyb_refl_coeffs, sym_set_labels);
  std::cout << "F_quartet = " << F_quartet.F_dag_bars[0].get_block(0)(0, 10, _, _) << std::endl;
}

TEST(Backbone, OCA) {
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // generate creation/annihilation operators
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order, Fq] =
     two_band_discrete_bath_helper_sym(beta, Lambda, eps);

  // set up backbone
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n                       = 4;
  auto B                      = Backbone(topology, n);

  // block-sparse diagram evaluation
  auto sym_set_labels = nda::zeros<long>(n);
  DiagramBlockSparseEvaluator D(beta, itops, Deltat, Deltat_refl, Gt, Fq, sym_set_labels);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_diagram_block_sparse(B);
  auto end                              = std::chrono::high_resolution_clock::now();
  auto OCA_result                       = D.Sigma;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "OCA (block-sparse) elapsed time: " << elapsed.count() << " s" << std::endl;

  // dense diagram evaluation
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);
  DiagramEvaluator D2(beta, itops, Deltat, Deltat_refl, Gt_dense, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_diagram_dense(B);
  end                   = std::chrono::high_resolution_clock::now();
  auto OCA_dense_result = D2.Sigma;
  elapsed               = end - start;
  std::cout << "OCA (dense) elapsed time: " << elapsed.count() << " s" << std::endl;

  // use block-sparse solver but with trivial sparsity
  std::vector<nda::array<dcomplex, 3>> Gt_dense_vec{Gt_dense};
  nda::vector<int> triv_bi{0};
  BlockDiagOpFun Gt_triv(Gt_dense_vec, triv_bi);

  std::vector<nda::array<dcomplex, 3>> Fs_dense_vec{Fs_dense};
  auto F_sym_triv = BlockOpSymSet(triv_bi, Fs_dense_vec);
  std::vector<nda::array<dcomplex, 3>> F_dags_dense_vec{F_dags_dense};
  auto F_dag_sym_triv = BlockOpSymSet(triv_bi, F_dags_dense_vec);
  auto Fq_triv        = BlockOpSymQuartet({F_sym_triv}, {F_dag_sym_triv}, hyb_coeffs, hyb_refl_coeffs, sym_set_labels);

  DiagramBlockSparseEvaluator D3(beta, itops, Deltat, Deltat_refl, Gt_triv, Fq_triv, sym_set_labels);
  start = std::chrono::high_resolution_clock::now();
  D3.eval_diagram_block_sparse(B);
  end                 = std::chrono::high_resolution_clock::now();
  auto OCA_trivial_bs = D3.Sigma;
  elapsed             = end - start;
  std::cout << "OCA (block-sparse, trivial) elapsed time: " << elapsed.count() << " s" << std::endl;

  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(0) - OCA_dense_result(_, range(0, 4), range(0, 4)))), 5e-16);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(1) - OCA_dense_result(_, range(4, 10), range(4, 10)))), 5e-16);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(2) - OCA_dense_result(_, range(10, 11), range(10, 11)))), 5e-16);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(3) - OCA_dense_result(_, range(11, 15), range(11, 15)))), 5e-16);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(4) - OCA_dense_result(_, range(15, 16), range(15, 16)))), 5e-16);

  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(0) - OCA_trivial_bs.get_block(0)(_, range(0, 4), range(0, 4)))), 5e-16);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(1) - OCA_trivial_bs.get_block(0)(_, range(4, 10), range(4, 10)))), 5e-16);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(2) - OCA_trivial_bs.get_block(0)(_, range(10, 11), range(10, 11)))), 5e-16);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(3) - OCA_trivial_bs.get_block(0)(_, range(11, 15), range(11, 15)))), 5e-16);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(4) - OCA_trivial_bs.get_block(0)(_, range(15, 16), range(15, 16)))), 5e-16);
}

TEST(Backbone, third_order) {
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // generate creation/annihilation operators
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order, Fq] =
     two_band_discrete_bath_helper_sym(beta, Lambda, eps);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  int n                       = 4;
  auto B                      = Backbone(topology, n);
  auto sym_set_labels         = nda::zeros<long>(n); // for block-sparse symmetries
  DiagramBlockSparseEvaluator D(beta, itops, Deltat, Deltat_refl, Gt, Fq, sym_set_labels);

  D.eval_diagram_block_sparse(B);
  auto third_result = D.Sigma;

  // compare to dense result
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);
  DiagramEvaluator D2(beta, itops, Deltat, Deltat_refl, Gt_dense, Fset);

  D2.eval_diagram_dense(B);
  auto third_result_dense = D2.Sigma;

  std::cout << third_result.get_block(0)(10, _, _) << std::endl;
  std::cout << third_result_dense(10, _, _) << std::endl;
}

// option to turn off symmetries (i.e. trivial block-sparsity), compare this to actual dense code
// OCA timing comparison
// crank up number of orbitals for spin flip fermion, compare time between dense and block-sparse solvers, plot, maybe play with parameters (e.g. beta), single-core

TEST(Backbone, spin_flip_fermion) {
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  std::string filename          = "../test/c++/h5/spin_flip_fermion.h5";
  int n                         = 4; // 2 * number of orbitals
  auto [hyb, hyb_refl]          = discrete_bath_spin_flip_helper(beta, Lambda, eps, n);
  auto hyb_coeffs               = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl_coeffs          = itops.vals2coefs(hyb_refl);
  auto [Gt, Fq, sym_set_labels] = load_from_hdf5(filename, beta, Lambda, eps, hyb_coeffs, hyb_refl_coeffs);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  DiagramBlockSparseEvaluator D(beta, itops, hyb, hyb_refl, Gt, Fq, sym_set_labels);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_diagram_block_sparse(B);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << std::setprecision(16) << "Block-sparse OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
  auto result = D.Sigma;

  // compare to dense result
  auto [Gt_dense, Fset, subspaces] =
     spin_flip_fermion_dense_helper(beta, Lambda, eps, hyb_coeffs, hyb_refl_coeffs, "../test/c++/h5/spin_flip_fermion.h5");

  DiagramEvaluator D2(beta, itops, hyb, hyb_refl, Gt_dense, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_diagram_dense(B);
  end      = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Dense OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
  auto result_dense = D2.Sigma;

  // use block-sparse solver but with trivial sparsity
  std::vector<nda::array<dcomplex, 3>> Gt_dense_vec{Gt_dense};
  nda::vector<int> triv_bi{0};
  BlockDiagOpFun Gt_triv(Gt_dense_vec, triv_bi);
  std::vector<nda::array<dcomplex, 3>> Fs_dense_vec{Fset.Fs};
  auto F_sym_triv = BlockOpSymSet(triv_bi, Fs_dense_vec);
  std::vector<nda::array<dcomplex, 3>> F_dags_dense_vec{Fset.F_dags};
  auto F_dag_sym_triv = BlockOpSymSet(triv_bi, F_dags_dense_vec);
  nda::vector<long> sym_set_labels_triv(n);
  sym_set_labels_triv = 0;
  auto Fq_triv        = BlockOpSymQuartet({F_sym_triv}, {F_dag_sym_triv}, hyb_coeffs, hyb_refl_coeffs, sym_set_labels_triv);
  DiagramBlockSparseEvaluator D3(beta, itops, hyb, hyb_refl, Gt_triv, Fq_triv, sym_set_labels);
  start = std::chrono::high_resolution_clock::now();
  D3.eval_diagram_block_sparse(B);
  end      = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Block-sparse trivial OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
  auto result_trivial_bs = D3.Sigma;

  ASSERT_LE(nda::max_element(nda::abs(result_dense - result_trivial_bs.get_block(0))), 1e-10);
  int i = 0, s0 = 0, s1 = 0;
  for (nda::vector_view<int> subspace : subspaces) {
    s1 += subspace.size();
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense(_, range(s0, s1), range(s0, s1)))), 1e-10);
    i += 1;
    s0 = s1;
  }
}

TEST(Backbone, spin_flip_fermion_sym_sets) {
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  std::string filename          = "../test/c++/h5/spin_flip_fermion_all_sym.h5";
  int n                         = 4; // 2 * number of orbitals
  auto [hyb, hyb_refl]          = discrete_bath_spin_flip_helper(beta, Lambda, eps, n);
  auto hyb_coeffs               = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl_coeffs          = itops.vals2coefs(hyb_refl);
  auto [Gt, Fq, sym_set_labels] = load_from_hdf5(filename, beta, Lambda, eps, hyb_coeffs, hyb_refl_coeffs);
  std::cout << std::setprecision(16) << std::endl;

  std::cout << "Fq size = " << Fq.Fs.size() << std::endl;

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  DiagramBlockSparseEvaluator D(beta, itops, hyb, hyb_refl, Gt, Fq, sym_set_labels);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_diagram_block_sparse(B);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << std::setprecision(16) << "Block-sparse OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
  auto result = D.Sigma;

  // compare to dense result
  auto [Gt_dense, Fset, subspaces] =
     spin_flip_fermion_dense_helper(beta, Lambda, eps, hyb_coeffs, hyb_refl_coeffs, "../test/c++/h5/spin_flip_fermion_all_sym.h5");

  DiagramEvaluator D2(beta, itops, hyb, hyb_refl, Gt_dense, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_diagram_dense(B);
  end      = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Dense OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
  auto result_dense = D2.Sigma;

  // use block-sparse solver but with trivial sparsity
  std::vector<nda::array<dcomplex, 3>> Gt_dense_vec{Gt_dense};
  nda::vector<int> triv_bi{0};
  BlockDiagOpFun Gt_triv(Gt_dense_vec, triv_bi);
  std::vector<nda::array<dcomplex, 3>> Fs_dense_vec{Fset.Fs};
  auto F_sym_triv = BlockOpSymSet(triv_bi, Fs_dense_vec);
  std::vector<nda::array<dcomplex, 3>> F_dags_dense_vec{Fset.F_dags};
  auto F_dag_sym_triv = BlockOpSymSet(triv_bi, F_dags_dense_vec);
  nda::vector<long> sym_set_labels_triv(n);
  sym_set_labels_triv = 0;
  auto Fq_triv        = BlockOpSymQuartet({F_sym_triv}, {F_dag_sym_triv}, hyb_coeffs, hyb_refl_coeffs, sym_set_labels_triv);
  DiagramBlockSparseEvaluator D3(beta, itops, hyb, hyb_refl, Gt_triv, Fq_triv, sym_set_labels_triv);
  start = std::chrono::high_resolution_clock::now();
  D3.eval_diagram_block_sparse(B);
  end      = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Block-sparse trivial OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
  auto result_trivial_bs = D3.Sigma;

  ASSERT_LE(nda::max_element(nda::abs(result_dense - result_trivial_bs.get_block(0))), 1e-10);
  int i = 0, s0 = 0, s1 = 0;
  for (nda::vector_view<int> subspace : subspaces) {
    s1 += subspace.size();
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense(_, range(s0, s1), range(s0, s1)))), 1e-10);
    i += 1;
    s0 = s1;
  }
}