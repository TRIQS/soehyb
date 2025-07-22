#include "triqs_soehyb/backbone.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <cppdlr/utils.hpp>
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
  T     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  GKt   = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Tkaps = nda::zeros<dcomplex>(n, r, Nmax, Nmax);
  Tmu   = nda::zeros<dcomplex>(r, Nmax, Nmax);
}

void DiagramBlockSparseEvaluator::multiply_vertex_block(Backbone &backbone, int v_ix, nda::vector_const_view<int> ind_path,
                                                        nda::vector_const_view<int> block_dims) {
  int o_ix    = backbone.get_vertex_orb(v_ix); // orbital_index
  int l_ix    = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  int n_col_r = v_ix < backbone.get_topology(0, 1) ? block_dims(1) : block_dims(0);
  int b_ix    = ind_path(v_ix - 1); // block index for the vertex v_ix

  if (backbone.has_vertex_bar(v_ix)) {   // F has bar
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        // TODO deal with more than one symmetry set, i.e., the [0] should actually be a loop over symmetry sets
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           nda::matmul(Fq.F_dag_bars[0].get_block(b_ix)(o_ix, l_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    } else {
      for (int t = 0; t < r; t++) {
        // TODO deal with more than one symmetry set, i.e., the [0] should actually be a loop over symmetry sets
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           nda::matmul(Fq.F_bars_refl[0].get_block(b_ix)(o_ix, l_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    }
  } else {
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        // TODO deal with more than one symmetry set, i.e., the [0] should actually be a loop over symmetry sets
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           nda::matmul(Fq.F_dags[0].get_block(b_ix)(o_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    } else {
      for (int t = 0; t < r; t++) {
        // TODO deal with more than one symmetry set, i.e., the [0] should actually be a loop over symmetry sets
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           nda::matmul(Fq.Fs[0].get_block(b_ix)(o_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    }
  }

  // K factor
  int bv      = backbone.get_vertex_Ksign(v_ix); // sign on K
  double pole = dlr_rf(l_ix);
  if (bv != 0) {
    for (int t = 0; t < r; t++) {
      T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) = k_it(dlr_it(t), bv * pole) * T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r));
    }
  }
}

void DiagramBlockSparseEvaluator::compose_with_edge_block(Backbone &backbone, int e_ix, nda::vector_const_view<int> ind_path,
                                                          nda::vector_const_view<int> block_dims) {

  int n_col_r = e_ix < backbone.get_topology(0, 1) ? block_dims(1) : block_dims(0);
  int b_ix    = ind_path(e_ix); // block index for the edge e_ix
  // TODO check block dims
  GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1))) = Gt.get_block(b_ix);
  int m                                                                  = backbone.m;
  for (int x = 0; x < m - 1; x++) {
    int be = backbone.get_edge(e_ix, x); // sign on K
    if (be != 0) {
      for (int t = 0; t < r; t++) {
        GKt(t, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1))) =
           k_it(dlr_it(t), be * dlr_rf(backbone.get_pole_ind(x))) * GKt(t, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)));
      }
    }
  }
  T(_, range(0, block_dims(e_ix + 1)), range(0, n_col_r)) =
     itops.convolve(beta, Fermion, itops.vals2coefs(GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)))),
                    itops.vals2coefs(T(_, range(0, block_dims(e_ix + 1)), range(0, n_col_r))), TIME_ORDERED);
}

void DiagramBlockSparseEvaluator::multiply_zero_vertex_block(Backbone &backbone, bool is_forward, int b_ix_0, nda::vector_const_view<int> ind_path,
                                                             nda::vector_const_view<int> block_dims) {

  int b_ix_mu = ind_path(backbone.get_topology(0, 1) - 1); // block index for F_mu
  int n       = backbone.n;
  if (is_forward) {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) {
        Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) =
           nda::matmul(T(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(1))), Fq.Fs[0].get_block(b_ix_0)(kap, _, _));
      }
    }
    T = 0;
    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) {
          Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) +=
             hyb(t, mu, kap) * Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0)));
        }
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(0))) +=
           nda::matmul(Fq.F_dags[0].get_block(b_ix_mu)(mu, _, _), Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))));
      }
    }
  } else {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) {
        Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) =
           nda::matmul(T(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(1))), Fq.F_dags[0].get_block(b_ix_0)(kap, _, _));
      }
    }
    T = 0;
    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) {
          Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) +=
             hyb_refl(t, mu, kap) * Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0)));
        }
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(0))) +=
           nda::matmul(Fq.Fs[0].get_block(b_ix_mu)(mu, _, _), Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))));
      }
    }
  }
}

void DiagramBlockSparseEvaluator::reset() {
  T     = 0;
  GKt   = 0;
  Tkaps = 0;
  Tmu   = 0;

  // TODO reset Sigma?
}

void DiagramBlockSparseEvaluator::eval_diagram_block_sparse(Backbone &backbone) {

  int m = backbone.m;
  nda::vector<int> ind_path(2 * m - 1);   // tracks block indices of factors for computing a particular block of the self-energy
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors
  // loop over all flat indices
  int f_ix_max = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(dlr_rf.size(), m - 1));
  for (int f_ix = 0; f_ix < f_ix_max; f_ix++) {
    backbone.set_flat_index(f_ix, dlr_rf); // set directions, pole indices, and orbital indices from a single integer index
    /*  Example of block_dims for m = 2 (OCA): each number is an index of block_dims, and each square represents a block of a matrix 
            3            3            2            2            1            1            0
        --------     --------     --------     --------     --------     --------     --------
      4 |   F  |   3 |   G  |   3 |   F  |   2 |   G  |   2 |   F  |   1 |   G  |   1 |   F  |
        |      |     |      |     |      |     |      |     |      |     |      |     |      |
        --------     --------     --------     --------     --------     --------     --------
    */
    for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); b_ix++) { // loop over blocks of self-energy
      bool path_all_nonzero = true;
      int w = 0, ip = b_ix; // w loops over the vertices and edges, ip is the current block index
      block_dims(0) = (backbone.has_vertex_dag(0)) ? Fq.F_dags[0].get_block_size(b_ix, 1) : Fq.Fs[0].get_block_size(b_ix, 1);
      block_dims(1) = (backbone.has_vertex_dag(0)) ? Fq.F_dags[0].get_block_size(b_ix, 0) : Fq.Fs[0].get_block_size(b_ix, 0);
      while (w < 2 * m && path_all_nonzero) { // only continue if we have not hit a zero block
        ip = (backbone.has_vertex_dag(w)) ? Fq.F_dags[0].get_block_index(ip) : Fq.Fs[0].get_block_index(ip); // update block index
        if (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)) { // check if we hit a zero block in F or Gt
          path_all_nonzero = false;
        } else {
          if (w < 2 * m - 1) {
            ind_path(w)       = ip; // store the block index for the current vertex/edge, unless we are at the last vertex
            block_dims(w + 2) = (backbone.has_vertex_dag(w + 1)) ? Fq.F_dags[0].get_block_size(ip, 0) : Fq.Fs[0].get_block_size(ip, 0);
          }
        }
        w += 1;
      }
      if (path_all_nonzero) {
        // evaluate the diagram with these directions, poles, and orbital indices
        // b_ix is the block index for the first edge
        eval_backbone_fixed_indices_block_sparse(backbone, b_ix, ind_path, block_dims);
      }
    }
    backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration
  }
}

void DiagramBlockSparseEvaluator::eval_backbone_fixed_indices_block_sparse(Backbone &backbone, int b_ix, nda::vector_const_view<int> ind_path,
                                                                           nda::vector_const_view<int> block_dims) {

  int m = backbone.m;

  T(_, range(0, block_dims(1)), range(0, block_dims(1))) = Gt.get_block(ind_path(0));
  for (int v = 1; v < backbone.get_topology(0, 1); v++) {
    multiply_vertex_block(backbone, v, ind_path, block_dims);
    compose_with_edge_block(backbone, v, ind_path, block_dims);
  }

  // TODO check b_ixs and block_dims_zero
  nda::vector<int> b_ixs           = {b_ix, ind_path(backbone.get_topology(0, 1) - 1)}; // block indices for F_kap and F_mu
  nda::vector<int> block_dims_zero = {block_dims(0), block_dims(1), block_dims(backbone.get_topology(0, 1)),
                                      block_dims(backbone.get_topology(0, 1) + 1)};
  multiply_zero_vertex_block(backbone, (not backbone.has_vertex_dag(0)), b_ix, ind_path, block_dims);

  for (int v = backbone.get_topology(0, 1) + 1; v < 2 * m; v++) {
    compose_with_edge_block(backbone, v - 1, ind_path, block_dims);
    multiply_vertex_block(backbone, v, ind_path, block_dims);
  }

  for (int p = 0; p < m - 1; p++) {
    int exp = backbone.get_prefactor_Kexp(p);
    if (exp != 0) {
      int Ksign = backbone.get_prefactor_Ksign(p);
      double om = dlr_rf(backbone.get_pole_ind(p));
      double k  = k_it(0, Ksign * om);
      for (int x = 0; x < exp; x++) T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))) /= k;
    }
  }
  int diag_order_sign = (m % 2 == 0) ? -1 : 1;
  T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))) *= diag_order_sign * backbone.prefactor_sign;
  Sigma.add_block(b_ix, T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))));
}