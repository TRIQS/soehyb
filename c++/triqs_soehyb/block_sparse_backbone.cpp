#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <iostream>
#include <nda/declarations.hpp>
#include <nda/nda.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_backbone.hpp>

DiagramBlockSparseEvaluator::DiagramBlockSparseEvaluator(double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                                         nda::array_const_view<dcomplex, 3> hyb_refl, BlockDiagOpFun &Gt, BlockOpSymQuartet &Fq)
   : beta(beta),
     itops(itops),
     r(itops.rank()),
     n(hyb.extent(1)),
     Nmax(Gt.get_max_block_size()),
     hyb(hyb),
     hyb_refl(hyb_refl),
     Gt(Gt),
     Fq(Fq),
     Sigma(itops.rank(), Gt.get_block_sizes()) {

  dlr_it = itops.get_itnodes();
  dlr_rf = itops.get_rfnodes();

  // allocate arrays
  T       = nda::zeros<dcomplex>(r, Nmax, Nmax);
  GKt     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Tkaps   = nda::zeros<dcomplex>(n, r, Nmax, Nmax);
  Tmu     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Sigma_L = nda::zeros<dcomplex>(r, Nmax, Nmax);
}

void DiagramBlockSparseEvaluator::multiply_vertex_block(Backbone &backbone, int v_ix, int b_ix, nda::vector_const_view<int> block_dims) {
  int o_ix = backbone.get_vertex_orb(v_ix); // orbital_index
  int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));

  if (backbone.has_vertex_bar(v_ix)) {   // F has bar
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        // TODO deal with more than one symmetry set, i.e., the [0] should actually be a loop over symmetry sets
        T(t, range(0, block_dims(1)), range(0, block_dims(0))) =
           nda::matmul(Fq.F_dag_bars[0].get_block(b_ix)(o_ix, l_ix, _, _), T(t, range(0, block_dims(0)), range(0, block_dims(0))));
      }
    } else {
      for (int t = 0; t < r; t++) {
        // TODO deal with more than one symmetry set, i.e., the [0] should actually be a loop over symmetry sets
        T(t, range(0, block_dims(1)), range(0, block_dims(0))) =
           nda::matmul(Fq.F_bars_refl[0].get_block(b_ix)(o_ix, l_ix, _, _), T(t, range(0, block_dims(0)), range(0, block_dims(0))));
      }
    }
  } else {
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        // TODO deal with more than one symmetry set, i.e., the [0] should actually be a loop over symmetry sets
        T(t, range(0, block_dims(1)), range(0, block_dims(0))) =
           nda::matmul(Fq.F_dags[0].get_block(b_ix)(o_ix, _, _), T(t, range(0, block_dims(0)), range(0, block_dims(0))));
      }
    } else {
      for (int t = 0; t < r; t++) {
        // TODO deal with more than one symmetry set, i.e., the [0] should actually be a loop over symmetry sets
        T(t, range(0, block_dims(1)), range(0, block_dims(0))) =
           nda::matmul(Fq.Fs[0].get_block(b_ix)(o_ix, _, _), T(t, range(0, block_dims(0)), range(0, block_dims(0))));
      }
    }
  }

  // K factor
  int bv      = backbone.get_vertex_Ksign(v_ix); // sign on K
  double pole = dlr_rf(l_ix);
  if (bv != 0) {
    for (int t = 0; t < r; t++) {
      T(t, range(0, block_dims(1)), range(0, block_dims(0))) = k_it(dlr_it(t), bv * pole) * T(t, range(0, block_dims(1)), range(0, block_dims(0)));
    }
  }
}

void DiagramBlockSparseEvaluator::compose_with_edge_block(Backbone &backbone, int e_ix, int b_ix, nda::vector_const_view<int> block_dims) {

  // TODO check block dims
  GKt(_, range(0, block_dims(1)), range(0, block_dims(1))) = Gt.get_block(b_ix);
  int m                                                    = backbone.m;
  for (int x = 0; x < m - 1; x++) {
    int be = backbone.get_edge(e_ix, x); // sign on K
    if (be != 0) {
      for (int t = 0; t < r; t++) {
        GKt(t, range(0, block_dims(1)), range(0, block_dims(1))) =
           k_it(dlr_it(t), be * dlr_rf(backbone.get_pole_ind(x))) * GKt(t, range(0, block_dims(1)), range(0, block_dims(1)));
      }
    }
  }
  T(_, range(0, block_dims(1)), range(0, block_dims(0))) =
     itops.convolve(beta, Fermion, itops.vals2coefs(GKt(_, range(0, block_dims(1)), range(0, block_dims(1)))),
                    itops.vals2coefs(T(_, range(0, block_dims(1)), range(0, block_dims(0)))), TIME_ORDERED);
}

void DiagramBlockSparseEvaluator::multiply_zero_vertex_block(Backbone &backbone, bool is_forward, nda::vector_const_view<int> b_ixs, nda::vector_const_view<int> block_dims) {

  // b_ixs should have two entries, {b_ix_kap, b_ix_mu}, the relevant block indices for Fkap and Fmu, resp. 
  // block_dims should have four entries, {a, b, c, d}
  // if q is the size of the symmetry group F_kap is q x b x a, T_kap is q x r x c x a, T_mu is r x c x a, T at the end is r x d x a
  int n = backbone.n;
  if (is_forward) {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) {
        Tkaps(kap, t, range(0, block_dims(2)), range(0, block_dims(0))) =
           nda::matmul(T(t, range(0, block_dims(2)), range(0, block_dims(1))), Fq.Fs[0].get_block(b_ixs(0))(kap, _, _));
      }
    }
    T = 0;
    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) {
          Tmu(t, range(0, block_dims(2)), range(0, block_dims(0))) +=
             hyb(t, mu, kap) * Tkaps(kap, t, range(0, block_dims(2)), range(0, block_dims(0)));
        }
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(3)), range(0, block_dims(0))) +=
           nda::matmul(Fq.F_dags[0].get_block(b_ixs(1))(mu, _, _), Tmu(t, range(0, block_dims(2)), range(0, block_dims(0))));
      }
    }
  } else {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) {
        Tkaps(kap, t, range(0, block_dims(2)), range(0, block_dims(0))) =
           nda::matmul(T(t, range(0, block_dims(2)), range(0, block_dims(1))), Fq.F_dags[0].get_block(b_ixs(0))(kap, _, _));
      }
    }
    T = 0;
    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) {
          Tmu(t, range(0, block_dims(2)), range(0, block_dims(0))) +=
             hyb_refl(t, mu, kap) * Tkaps(kap, t, range(0, block_dims(2)), range(0, block_dims(0)));
        }
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(3)), range(0, block_dims(0))) +=
           nda::matmul(Fq.Fs[0].get_block(b_ixs(1))(mu, _, _), Tmu(t, range(0, block_dims(2)), range(0, block_dims(0))));
      }
    }
  }
}