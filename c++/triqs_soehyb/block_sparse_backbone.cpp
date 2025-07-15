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
  }
}

BlockDiagOpFun eval_backbone(Backbone &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb, BlockDiagOpFun &Gt,
                             std::vector<BlockOp> &Fs, std::vector<BlockOp> &F_dags) {
  nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
  nda::vector_const_view<double> dlr_it = itops.get_itnodes();
  // number of imaginary time nodes
  int r = dlr_it.extent(0);
  int m = backbone.m;

  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(hyb));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  int n                = Fs[0].get_block(0).extent(0);

  // compute Fbars and Fdagbars
  auto Fbar_indices    = Fs[0].get_block_indices();
  auto Fbar_sizes      = Fs[0].get_block_sizes();
  auto Fdagbar_indices = F_dags[0].get_block_indices();
  auto Fdagbar_sizes   = F_dags[0].get_block_sizes();
  std::vector<std::vector<BlockOp>> Fdagbars(n, std::vector<BlockOp>(r, BlockOp(Fdagbar_indices, Fdagbar_sizes)));
  std::vector<std::vector<BlockOp>> Fbarsrefl(n, std::vector<BlockOp>(r, BlockOp(Fbar_indices, Fbar_sizes)));
  for (int lam = 0; lam < n; lam++) {
    for (int l = 0; l < r; l++) {
      for (int nu = 0; nu < n; nu++) {
        Fdagbars[lam][l] += hyb_coeffs(l, nu, lam) * F_dags[nu];
        Fbarsrefl[lam][l] += hyb_refl_coeffs(l, lam, nu) * Fs[nu];
      }
    }
  }

  // initialize self-energy
  BlockDiagOpFun Sigma = BlockDiagOpFun(r, Gt.get_block_sizes());
  int bc               = Gt.get_num_block_cols();

  // preallocate intermediate arrays
  // TODO for b-s version, initialize largest needed array and write into top-left corner
  bool path_all_nonzero = true;
  nda::vector<int> ind_path(2 * m - 1);
  nda::vector<int> block_dims(2 * m + 1);
  int Nmax = Gt.get_max_block_size();
  nda::array<dcomplex, 3> Sigma_L(r, Nmax, Nmax), T(r, Nmax, Nmax), Tmu(r, Nmax, Nmax), GKt(r, Nmax, Nmax);
  nda::array<dcomplex, 4> Tkaps(n, r, Nmax, Nmax);
  // Sigma_l = term of self-energy assoc'd with pole l, rest are placeholders

  // loop over hybridization lines
  nda::vector<int> fb_vec(m), orb_inds(2 * m);
  auto pole_inds = nda::zeros<int>(m - 1);

  for (int fb = 0; fb < pow(2, m); fb++) { // loop over 2^m combos of for/backward lines
    int fb0 = fb;
    // turn (int) fb into a vector of 1s and 0s corresp. to forward, backward lines, resp.
    for (int i = 0; i < m; i++) {
      fb_vec(i) = fb0 % 2;
      fb0       = fb0 / 2;
    }
    backbone.set_directions(fb_vec);             // give line directions to backbone object
    int sign = (fb_vec(0) ^ fb_vec(1)) ? -1 : 1; // TODO fix

    for (int b = 0; b < bc; b++) { // loop over blocks of self-energy
      // "backwards pass"
      //
      // for each self-energy block, find contributing blocks of factors
      //
      // paths_all_nonzero: false if for i-th block of
      // Sigma, factors assoc'd with lambda, mu, kappa don't contribute
      //
      // ind_path: a vector of column indices of the
      // blocks of the factors that contribute. if paths_all_nonzero is
      // false at this index, values in ind_path are garbage.
      //
      // ATTN: assumes all BlockOps in F(1,2,3)list have the same structure
      // i.e, index path is independent of kappa, mu
      path_all_nonzero = true;
      int w = 0, ip = b;
      block_dims(0) = (backbone.has_vertex_dag(0)) ? F_dags[0].get_block_size(b, 1) : Fs[0].get_block_size(b, 1);
      // block_dims(1) = (backbone.vertex_dags(0)) ? F_dags[0].get_block_size(b,0) : Fs[0].get_block_size(b,0);
      while (w < 2 * m && path_all_nonzero) { // loop over vertices
        ip = (backbone.has_vertex_dag(w)) ? F_dags[0].get_block_index(ip) : Fs[0].get_block_index(ip);
        if (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)) {
          path_all_nonzero = false;
        } else {
          if (w < 2 * m - 1) ind_path(w) = ip;
          block_dims(w + 1) = (backbone.has_vertex_dag(w)) ? F_dags[0].get_block_size(ip, 0) : Fs[0].get_block_size(ip, 0);
        }
        w += 1;
      }

      if (path_all_nonzero) {
        auto Sigma_b = nda::make_regular(0 * Sigma.get_block(b));
        // L = pole multiindex
        for (int L = 0; L < pow(r, m - 1); L++) { // loop over all combinations of pole indices
          Sigma_L = 0;
          int L0  = L;
          // turn (int) L into a vector of pole indices
          for (int i = 0; i < m - 1; i++) {
            pole_inds(i) = L0 % r;
            L0           = L0 / r;
          }
          backbone.set_pole_inds(pole_inds, dlr_rf); // give pole indices to backbone object

          // print a negative and positive pole diagram for each set of line directions
          // if (L == 0) std::cout << "fb = " << fb << " " << backbone << std::endl;
          // if (L == r-1) std::cout << "fb = " << fb << " " << backbone << std::endl;

          // set up orbital (Greek) indices that are explicity summed over
          for (int s = 0; s < pow(n, m - 1); s++) { // loop over all combos of orbital indices
            int s0 = s;
            // turn (int) s into a vector of orbital indices
            for (int i = 1; i < m; i++) { // loop over lines, skipping the one connected to zero
              orb_inds(backbone.get_topology(i, 0)) = s0 % n;
              orb_inds(backbone.get_topology(i, 1)) = s0 % n;
              // orbital indices on vertices connected by a line are the same
              s0 = s0 / n;
            }
            backbone.set_orb_inds(orb_inds);

            // 1. Starting from tau_1, proceed right to left, performing
            //    multiplications at vertices and convolutions at edges,
            //    until reaching the vertex containing the undecomposed
            //    hybridization line Delta_{mu kappa}
            T                                                      = 0;
            T(_, range(0, block_dims(0)), range(0, block_dims(1))) = Gt.get_block(b); // T stores the result moving right to left
            // T is initialized with block b of Gt, which is always the function at the rightmost edge

            // !!!!!!!!!!!!!!!!
            // TODO: block_dims
            // !!!!!!!!!!!!!!!!

            /*
                        for (int v = 1; v < backbone.get_topology(0,1); v++) { // loop from the first vertex to before the special vertex
                            multiply_vertex_block(backbone, dlr_it, dlr_rf, Fs, F_dags, Fdagbars, Fbarsrefl, v, ind_path(v), T, block_dims(range(v,v+1)));
                            compose_with_edge_block(backbone, itops, dlr_it, dlr_rf, beta, Gt, v, ind_path(v), T, GKt(_,range(0,block_dims(v+1)),range(0,block_dims(v+1))));
                        }

                        // 2. For each kappa, multiply by F_kappa(^dag). Then for each 
                        //    mu, kappa, multiply by Delta_{mu kappa}, and sum over 
                        //    kappa. Finally for each mu, multiply F_mu[^dag] and sum 
                        //    over mu. 
                        if (not backbone.has_vertex_dag(0)) { // line connected to zero is forward
                            multiply_zero_vertex(backbone, hyb, Fs, F_dags, Tkaps, Tmu, ind_path(backbone.get_topology(0,1)), T); 
                        } else { // line connected to zero is backward, NOTE NEGATIVE SIGN IN HYB
                            multiply_zero_vertex(backbone, nda::make_regular(-hyb_refl), F_dags, Fs, Tkaps, Tmu, ind_path(backbone.get_topology(0,1)), T); 
                        }

                        // 3. Continue right to left until the final vertex 
                        //    multiplication is complete.
                        for (int v = backbone.get_topology(0,1) + 1; v < 2*m; v++) { // loop from the next edge to the final vertex
                            compose_with_edge_block(backbone, itops, dlr_it, dlr_rf, beta, Gt, v-1, ind_path(v), T, GKt); 
                            multiply_vertex_block(backbone, dlr_it, dlr_rf, Fs, F_dags, Fdagbars, Fbarsrefl, v, ind_path(v), T, block_dims(range(v,v+1))); 
                        }
                        */
            Sigma_L += T;
            backbone.reset_orb_inds();
          } // sum over orbital indices

          // 4. Multiply by prefactor
          for (int p = 0; p < m - 1; p++) {           // loop over pole indices
            int exp = backbone.get_prefactor_Kexp(p); // exponent on K for this pole index
            if (exp != 0) {
              int Ksign = backbone.get_prefactor_Ksign(p);
              double om = dlr_rf(pole_inds(p));
              double k  = k_it(0, Ksign * om);
              for (int q = 0; q < exp; q++) Sigma_L = Sigma_L / k;
            }
          }
          Sigma_b += sign * backbone.prefactor_sign * Sigma_L; // TODO: block_sizes for Sigma_L too
          backbone.reset_pole_inds();
        } // sum over poles
        Sigma.add_block(b, Sigma_b);
      } // end if(path_all_nonzero)
    } // loop over blocks
    backbone.reset_directions();
  } // sum over forward/backward lines

  return Sigma;
}