#include <iomanip>
#include <iostream>
#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include "triqs_soehyb/dense_backbone.hpp"
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_manual.hpp>
#include <triqs_soehyb/backbone.hpp>
#include <triqs_soehyb/block_sparse_backbone.hpp>


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

  auto sym_set_labels = nda::zeros<long>(n);
  auto F_quartet      = BlockOpSymQuartet(F_sym_vec, F_dag_sym_vec, hyb_coeffs, hyb_refl_coeffs, sym_set_labels);
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
  DiagramBlockSparseEvaluator D(beta, itops, Deltat, Deltat_refl, Gt, Fq);
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

  DiagramBlockSparseEvaluator D3(beta, itops, Deltat, Deltat_refl, Gt_triv, Fq_triv);
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
  int n                         = 2 * 5; // 2 * number of orbitals
  auto [hyb, hyb_refl]          = discrete_bath_spin_flip_helper(beta, Lambda, eps, n);
  auto hyb_coeffs               = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl_coeffs          = itops.vals2coefs(hyb_refl);
  auto [Gt, Fq, sym_set_labels] = load_from_hdf5(filename, beta, Lambda, eps, hyb_coeffs, hyb_refl_coeffs);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  DiagramBlockSparseEvaluator D(beta, itops, hyb, hyb_refl, Gt, Fq);
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
  DiagramBlockSparseEvaluator D3(beta, itops, hyb, hyb_refl, Gt_triv, Fq_triv);
  start = std::chrono::high_resolution_clock::now();
  D3.eval_diagram_block_sparse(B);
  end      = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Block-sparse trivial OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
  auto result_trivial_bs = D3.Sigma;

  ASSERT_LE(nda::max_element(nda::abs(result_dense - result_trivial_bs.get_block(0))), 1e-10);
  int i = 0, s0 = 0, s1 = 0;
  for (nda::vector_view<unsigned long> subspace : subspaces) {
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
  int n                         = 2 * 5; // 2 * number of orbitals
  auto [hyb, hyb_refl]          = discrete_bath_spin_flip_helper(beta, Lambda, eps, n);
  auto hyb_coeffs               = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl_coeffs          = itops.vals2coefs(hyb_refl);
  auto [Gt, Fq, sym_set_labels] = load_from_hdf5(filename, beta, Lambda, eps, hyb_coeffs, hyb_refl_coeffs);
  
  std::cout << std::setprecision(16);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  DiagramBlockSparseEvaluator D(beta, itops, hyb, hyb_refl, Gt, Fq);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_diagram_block_sparse(B);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Block-sparse OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
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
  DiagramBlockSparseEvaluator D3(beta, itops, hyb, hyb_refl, Gt_triv, Fq_triv);
  start = std::chrono::high_resolution_clock::now();
  D3.eval_diagram_block_sparse(B);
  end      = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Block-sparse trivial OCA evaluation for n = " << n << " took " << duration.count() << " seconds." << std::endl;
  auto result_trivial_bs = D3.Sigma;

  ASSERT_LE(nda::max_element(nda::abs(result_dense - result_trivial_bs.get_block(0))), 1e-10);
  int i = 0, s0 = 0, s1 = 0;
  for (nda::vector_view<unsigned long> subspace : subspaces) {
    s1 += subspace.size();
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense(_, range(s0, s1), range(s0, s1)))), 1e-10);
    i += 1;
    s0 = s1;
  }
}

TEST(Backbone, PYTHON_third_order) {
  nda::array<int, 3> topologies = {{{0, 2}, {1, 4}, {3, 5}}, {{0, 3}, {1, 5}, {2, 4}}, {{0, 4}, {1, 3}, {2, 5}}, {{0, 3}, {1, 4}, {2, 5}}};
  nda::vector<int> topo_sign{1, 1, 1, -1}; // topo_sign(i) = (-1)^{# of line crossings in topology i}

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 10.0 * beta; // 1000.0*beta;
  double eps    = 1.0e-10;

  // generate creation/annihilation operators
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order, Fq] =
     two_band_discrete_bath_helper_sym(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // hybridization and DenseFSet
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // compute NCA and OCA
  auto NCA_result          = NCA_dense(Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense);
  nda::array<int, 2> T_OCA = {{0, 2}, {1, 3}};
  auto B_OCA               = Backbone(T_OCA, n);
  auto D                   = DiagramBlockSparseEvaluator(beta, itops, Deltat, Deltat_refl, Gt, Fq); // create DiagramEvaluator object
  D.eval_diagram_block_sparse(B_OCA);                                                                   // evaluate OCA diagram
  auto OCA_result = D.Sigma;                                                                     // get the result from the DiagramEvaluator
  D.reset();

  BlockDiagOpFun third_order_result(r, Gt.get_block_sizes()); 
  BlockDiagOpFun third_order_02_result(r, Gt.get_block_sizes()); 
  BlockDiagOpFun third_order_0314_result(r, Gt.get_block_sizes());
  BlockDiagOpFun third_order_0315_result(r, Gt.get_block_sizes());
  BlockDiagOpFun third_order_04_result(r, Gt.get_block_sizes());

  for (int i = 0; i < 4; i++) {
    auto B = Backbone(topologies(i, _, _), n);
    D.eval_diagram_block_sparse(B);
    third_order_result += topo_sign(i) * D.Sigma; // accumulate results with sign
    if (i == 0) {
      third_order_02_result = topo_sign(i) * D.Sigma; // store results for specific topologies
    } else if (i == 1) {
      third_order_0315_result = topo_sign(i) * D.Sigma;
    } else if (i == 2) {
      third_order_04_result = topo_sign(i) * D.Sigma;
    } else if (i == 3) {
      third_order_0314_result = topo_sign(i) * D.Sigma;
    }
    D.reset(); // reset the DiagramEvaluator for the next topology
  }
  
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

  int i = 0, s0 = 0, s1 = 0;
  for (auto subspace : subspaces) {
    s1 += subspace.size();
    ASSERT_LE(nda::max_element(nda::abs(third_order_result.get_block(i) - third_order_py_perm(_, range(s0, s1), range(s0, s1)))), 10 * eps);
    i += 1;
    s0 = s1;
  }
}