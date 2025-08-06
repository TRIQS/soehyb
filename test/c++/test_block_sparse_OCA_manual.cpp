#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_manual.hpp>
#include <triqs_soehyb/backbone.hpp>
#include <triqs_soehyb/dense_backbone.hpp>
#include <triqs_soehyb/block_sparse_backbone.hpp>

TEST(BlockSparseOCAManual, single_exponential) {
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

TEST(BlockSparseOCAManual, PYTHON_two_band_discrete_bath_bs) {
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

TEST(BlockSparseOCAManual, PYTHON_two_band_discrete_bath_dense) {
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

TEST(BlockSparseOCAManual, two_band_discrete_bath_bs_vs_dense) {
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

TEST(BlockSparseOCAManual, PYTHON_two_band_discrete_bath_tpz) {
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