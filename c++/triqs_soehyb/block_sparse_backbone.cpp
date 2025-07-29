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
                                                         nda::array_const_view<dcomplex, 3> hyb_refl, BlockDiagOpFun &Gt, BlockOpSymQuartet &Fq,
                                                         nda::vector_const_view<long> sym_set_labels)
   : beta(beta),
     itops(itops),
     r(itops.rank()),
     n(hyb.extent(1)),
     q(nda::max_element(sym_set_labels) + 1),
     Nmax(Gt.get_max_block_size()),
     hyb(hyb),
     hyb_refl(hyb_refl),
     Gt(Gt),
     Fq(Fq),
     Sigma(itops.rank(), Gt.get_block_sizes()),
     sym_set_labels(sym_set_labels) {

  dlr_it = itops.get_itnodes();
  dlr_rf = itops.get_rfnodes();

  // allocate arrays
  T     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  GKt   = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Tkaps = nda::zeros<dcomplex>(n, r, Nmax, Nmax);
  Tmu   = nda::zeros<dcomplex>(r, Nmax, Nmax);

  // calculate symmetry set indices
  sym_set_inds  = nda::zeros<long>(n);
  sym_set_sizes = nda::zeros<long>(q);
  for (int i = 0; i < n; i++) {
    sym_set_inds(i) = sym_set_sizes(sym_set_labels(i));
    sym_set_sizes(sym_set_labels(i))++;
  }
}

void DiagramBlockSparseEvaluator::multiply_vertex_block(Backbone &backbone, int v_ix, nda::vector_const_view<int> ind_path,
                                                        nda::vector_const_view<int> block_dims) {
  int o_ix = backbone.get_vertex_orb(v_ix); // orbital_index
  // split backbone orbital index into symmetry set index and orbital index within the symmetry set
  // i.e. have mapping between backbone orbital index and symmetry set index
  int q_ix    = static_cast<int>(sym_set_labels[o_ix]); // symmetry set index
  int qo_ix   = static_cast<int>(sym_set_inds(o_ix));   // index within the symmetry set
  int l_ix    = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  int n_col_r = v_ix < backbone.get_topology(0, 1) ? block_dims(1) : block_dims(0);
  int b_ix    = ind_path(v_ix - 1); // block index for the vertex v_ix

  if (backbone.has_vertex_bar(v_ix)) {   // F has bar
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      if (v_ix == 3) {
        std::cout << "Vertex 3: " << std::endl;
        std::cout << "F dimension: " << Fq.F_dag_bars[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _).shape() << std::endl;
        std::cout << "T dimensions: " << block_dims(v_ix) << " by " << n_col_r << std::endl;
        std::cout << "dag bar" << std::endl;
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           nda::matmul(Fq.F_dag_bars[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    } else {
      if (v_ix == 3) {
        std::cout << "Vertex 3: " << std::endl;
        std::cout << "F dimension: " << Fq.F_bars_refl[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _).shape() << std::endl;
        std::cout << "T dimensions: " << block_dims(v_ix) << " by " << n_col_r << std::endl;
        std::cout << "bar" << std::endl;
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           nda::matmul(Fq.F_bars_refl[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    }
  } else {
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           nda::matmul(Fq.F_dags[q_ix].get_block(b_ix)(qo_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    } else {
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           nda::matmul(Fq.Fs[q_ix].get_block(b_ix)(qo_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
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

void DiagramBlockSparseEvaluator::multiply_zero_vertex_block(Backbone &backbone, bool is_forward, int b_ix_0, int p_kap, int p_mu,
                                                             nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims) {

  int b_ix_mu = ind_path(backbone.get_topology(0, 1) - 1); // block index for F_mu
  if (is_forward) {
    for (int kap = 0; kap < sym_set_sizes(p_kap); kap++) {
      for (int t = 0; t < r; t++) {
        Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) =
           nda::matmul(T(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(1))), Fq.Fs[p_kap].get_block(b_ix_0)(kap, _, _));
      }
    }
    T = 0;
    for (int mu = 0; mu < sym_set_sizes(p_mu); mu++) {
      Tmu = 0;
      for (int kap = 0; kap < sym_set_sizes(p_kap); kap++) {
        for (int t = 0; t < r; t++) {
          Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) +=
             hyb(t, mu, kap) * Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0)));
        }
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(0))) += nda::matmul(
           Fq.F_dags[p_mu].get_block(b_ix_mu)(p_mu, _, _), Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))));
      }
    }
  } else {
    for (int kap = 0; kap < sym_set_sizes(p_kap); kap++) {
      for (int t = 0; t < r; t++) {
        Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) = nda::matmul(
           T(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(1))), Fq.F_dags[p_kap].get_block(b_ix_0)(kap, _, _));
      }
    }
    T = 0;
    for (int mu = 0; mu < sym_set_sizes(p_mu); mu++) {
      Tmu = 0;
      for (int kap = 0; kap < sym_set_sizes(p_kap); kap++) {
        for (int t = 0; t < r; t++) {
          Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) +=
             hyb_refl(t, mu, kap) * Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0)));
        }
      }

      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(0))) +=
           nda::matmul(Fq.Fs[p_mu].get_block(b_ix_mu)(mu, _, _), Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))));
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
  nda::vector<int> ind_path(2 * m - 1); // tracks block indices of factors for computing a particular block of the self-energy
  // nda::vector<int> ind_path(backbone.get_topology(0, 1) - 1);
  // ^ tracks block indices of factors for computing a particular block of the self-energy before the vertex connected to zero
  // nda::array<int, 2> ind_path_forks(2 * m - backbone.get_topology(0, 1), q);
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors
  // nda::vector<int> block_dims(backbone.get_topology(0, 1) + 1);
  // ^ tracks the dimensions of the blocks in these factors before the vertex connected to zero
  // nda::array<int, 2> block_dims_forks(2 * m - backbone.get_topology(0, 1), q);

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
    // ind_path can diverge at the vertex connected to vertex 0
    for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); b_ix++) { // loop over blocks of self-energy
      std::cout << "========== b_ix: " << b_ix << " ===========" << std::endl;
      for (int p_kap = 0; p_kap < q; p_kap++) {                  // loop over symmetry sets on the zero vertex
        std::cout << "------------ p_kap: " << p_kap << " --------" << std::endl;
        bool path_all_nonzero = true;
        int w = 0, ip = 0; // w loops over the vertices and edges, ip is the current block index

        if (backbone.has_vertex_dag(0)) { // if line is connected to zero is backward
          ip = Fq.F_dags[p_kap].get_block_index(b_ix);
          if (ip != -1) {
            block_dims(0) = Fq.F_dags[p_kap].get_block_size(b_ix, 1);
            block_dims(1) = Fq.F_dags[p_kap].get_block_size(b_ix, 0);
          } else {
            path_all_nonzero = false;
          }
        } else {
          ip = Fq.Fs[p_kap].get_block_index(b_ix);
          if (ip != -1) {
            block_dims(0) = Fq.Fs[p_kap].get_block_size(b_ix, 1);
            block_dims(1) = Fq.Fs[p_kap].get_block_size(b_ix, 0);
          } else {
            path_all_nonzero = false;
          }
        }
        /*
        block_dims(0) = (backbone.has_vertex_dag(0)) ? Fq.F_dags[p_kap].get_block_size(b_ix, 1) :
                                                       Fq.Fs[p_kap].get_block_size(b_ix, 1);
        block_dims(1) = (backbone.has_vertex_dag(0)) ? Fq.F_dags[p_kap].get_block_size(b_ix, 0) :
                                                       Fq.Fs[p_kap].get_block_size(b_ix, 0);
        */
        // traverse factors in two halves
        // first half: before vertex connected to zero
        while (w < backbone.get_topology(0, 1) && path_all_nonzero) { // only continue if we have not hit a zero block
          if (w != 0) {
            ip = (backbone.has_vertex_dag(w)) ? Fq.F_dags[sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip) :
                                                Fq.Fs[sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip); // update block index
          }
          std::cout << "w: " << w << ", ip: " << ip << std::endl;
          if (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)) { // check if we hit a zero block in F or Gt
            path_all_nonzero = false;
          } else {                                      // inner 'if' block unnecessary in first half
            ind_path(w) = ip;                           // store the block index for the current vertex/edge, unless we are at the last vertex
            if (w != backbone.get_topology(0, 1) - 1) { // if so, then orb_ind = -1, because w is the vertex connected to zero
              block_dims(w + 2) = (backbone.has_vertex_dag(w + 1)) ? Fq.F_dags[sym_set_labels(backbone.get_orb_ind(w + 1))].get_block_size(ip, 0) :
                                                                     Fq.Fs[sym_set_labels(backbone.get_orb_ind(w + 1))].get_block_size(ip, 0);
            }
          }
          w += 1;
        }

        int ip1 = ip; 
        // second half
        if (path_all_nonzero) {
          for (int p_mu = 0; p_mu < q; p_mu++) {
          bool fork_all_nonzero = true;
            std::cout << "------------ p_mu: " << p_mu << " --------" << std::endl;
            w = backbone.get_topology(0, 1); // reset w to the vertex connected to vertex 0
            // save block_dims(backbone.get_topology(0, 1) + 1)
            block_dims(w + 1) = (backbone.has_vertex_dag(w)) ? Fq.F_dags[p_mu].get_block_size(ip1, 0) : Fq.Fs[p_mu].get_block_size(ip1, 0);
            ip                = (backbone.has_vertex_dag(w)) ? Fq.F_dags[p_mu].get_block_index(ip1) :
                                                               Fq.Fs[p_mu].get_block_index(ip1); // update block index for the vertex connected to zero
            while (w < 2 * m && fork_all_nonzero) {
              if (w != backbone.get_topology(0, 1)) {
                ip = (backbone.has_vertex_dag(w)) ? Fq.F_dags[sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip) :
                                                    Fq.Fs[sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip); // update block index
              }
              std::cout << "w: " << w << ", ip: " << ip << std::endl;
              if (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)) { // check if we hit a zero block in F or Gt
                fork_all_nonzero = false;
              } else {
                if (w < 2 * m - 1) {      // only store the block index if we are not at the last vertex
                  ind_path(w)       = ip; // store the block index for the current vertex/edge
                  block_dims(w + 2) = (backbone.has_vertex_dag(w + 1)) ?
                     Fq.F_dags[sym_set_labels(backbone.get_orb_ind(w + 1))].get_block_size(ip, 0) :
                     Fq.Fs[sym_set_labels(backbone.get_orb_ind(w + 1))].get_block_size(ip, 0);
                }
              }
              std::cout << "fork_all_nonzero: " << fork_all_nonzero << std::endl;
              w += 1;
            }
            if (fork_all_nonzero) {
              // evaluate the diagram with these directions, poles, and orbital indices
              // b_ix is the block index for the first edge

              // print the blocks we'll multiply together
              for (int x = 0; x < 2 * m; x++) {
                std::cout << "x = " << x << std::endl;
                if (x == 0) {
                  if (backbone.has_vertex_dag(x)) {
                    std::cout << Fq.F_dags[p_kap].get_block(b_ix).shape() << std::endl;
                  } else {
                    std::cout << Fq.Fs[p_kap].get_block(b_ix).shape() << std::endl;
                  }
                } else if (x == backbone.get_topology(0, 1)) {
                  if (backbone.has_vertex_dag(x)) {
                    std::cout << Fq.F_dags[p_mu].get_block(ind_path(x - 1)).shape() << std::endl;
                    std::cout << "block index = " << Fq.F_dags[p_mu].get_block_index(ind_path(x - 1)) << std::endl;
                  } else {
                    std::cout << Fq.Fs[p_mu].get_block(ind_path(x - 1)).shape() << std::endl;
                    std::cout << "block index = " << Fq.Fs[p_mu].get_block_index(ind_path(x - 1)) << std::endl;
                  }
                } else {
                  if (backbone.has_vertex_dag(x)) {
                    std::cout << Fq.F_dags[sym_set_labels(backbone.get_orb_ind(x))].get_block(ind_path(x - 1)).shape() << std::endl;
                  } else {
                    std::cout << Fq.Fs[sym_set_labels(backbone.get_orb_ind(x))].get_block(ind_path(x - 1)).shape() << std::endl;
                  }
                }
              }
              std::cout << "block_dims = " << block_dims << std::endl;
              std::cout << "ind_path = " << ind_path << std::endl;
              eval_backbone_fixed_indices_block_sparse(backbone, b_ix, p_kap, p_mu, ind_path, block_dims);
            }
            std::cout << "------------------------" << std::endl;
          }
          std::cout << "p_mu loop finished" << std::endl;
        }
      }
      std::cout << "p_kap loop finished" << std::endl;

      // std::cout << backbone.get_orb_ind(0) << std::endl;
      // std::cout << Fq.F_dags[sym_set_labels(backbone.get_orb_ind(0))].get_block(b_ix) << std::endl;
      // std::cout << Fq.Fs[sym_set_labels(backbone.get_orb_ind(0))].get_block_size(b_ix, 1) << std::endl;
      // std::cout << Fq.F_dags[sym_set_labels(backbone.get_orb_ind(0))].get_block_size(b_ix, 0) << std::endl;
      // std::cout << Fq.Fs[sym_set_labels(backbone.get_orb_ind(0))].get_block_size(b_ix, 0) << std::endl;

      /*
      while (w < 2 * m && path_all_nonzero) { // only continue if we have not hit a zero block
        ip = (backbone.has_vertex_dag(w)) ? Fq.F_dags[backbone.get_orb_ind(w)].get_block_index(ip) :
                                            Fq.Fs[backbone.get_orb_ind(w)].get_block_index(ip); // update block index
        if (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)) {                 // check if we hit a zero block in F or Gt
          path_all_nonzero = false;
        } else {
          if (w < 2 * m - 1) {
            ind_path(w)       = ip; // store the block index for the current vertex/edge, unless we are at the last vertex
            block_dims(w + 2) = (backbone.has_vertex_dag(w + 1)) ? Fq.F_dags[backbone.get_orb_ind(w + 1)].get_block_size(ip, 0) :
                                                                   Fq.Fs[backbone.get_orb_ind(w + 1)].get_block_size(ip, 0);
          }
        }
        w += 1;
      }
      */
    }
    std::cout << "b_ix loop finished" << std::endl;
    backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration
  }
}

void DiagramBlockSparseEvaluator::eval_backbone_fixed_indices_block_sparse(Backbone &backbone, int b_ix, int p_kap, int p_mu,
                                                                           nda::vector_const_view<int> ind_path,
                                                                           nda::vector_const_view<int> block_dims) {

  int m = backbone.m;

  std::cout << "step 0" << std::endl;
  T(_, range(0, block_dims(1)), range(0, block_dims(1))) = Gt.get_block(ind_path(0));
  for (int v = 1; v < backbone.get_topology(0, 1); v++) {
    multiply_vertex_block(backbone, v, ind_path, block_dims);
    compose_with_edge_block(backbone, v, ind_path, block_dims);
  }
  std::cout << "step 1" << std::endl;

  nda::vector<int> b_ixs = {b_ix, ind_path(backbone.get_topology(0, 1) - 1)}; // block indices for F_kap and F_mu
  multiply_zero_vertex_block(backbone, (not backbone.has_vertex_dag(0)), b_ix, p_kap, p_mu, ind_path, block_dims);
  std::cout << "step 2" << std::endl;

  for (int v = backbone.get_topology(0, 1) + 1; v < 2 * m; v++) {
    compose_with_edge_block(backbone, v - 1, ind_path, block_dims);
    multiply_vertex_block(backbone, v, ind_path, block_dims);
  }
  std::cout << "step 3" << std::endl;

  for (int p = 0; p < m - 1; p++) {
    int exp = backbone.get_prefactor_Kexp(p);
    if (exp != 0) {
      int Ksign = backbone.get_prefactor_Ksign(p);
      double om = dlr_rf(backbone.get_pole_ind(p));
      double k  = k_it(0, Ksign * om);
      for (int x = 0; x < exp; x++) T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))) /= k;
    }
  }
  std::cout << "step 4" << std::endl;
  int diag_order_sign = (m % 2 == 0) ? -1 : 1;
  T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))) *= diag_order_sign * backbone.prefactor_sign;
  Sigma.add_block(b_ix, T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))));
}