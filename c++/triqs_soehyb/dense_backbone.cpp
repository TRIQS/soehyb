#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/nda.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/dense_backbone.hpp>

DiagramEvaluator::DiagramEvaluator(double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                   nda::array_const_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset)
   : beta(beta), itops(itops), hyb(hyb), hyb_refl(hyb_refl), Gt(Gt), Fset(Fset) {
  dlr_it = itops.get_itnodes();
  dlr_rf = itops.get_rfnodes();

  // allocate arrays
  int r   = dlr_it.extent(0);
  int n   = hyb.extent(1);
  int N   = Gt.extent(1);
  T       = nda::zeros<dcomplex>(r, N, N);
  GKt     = nda::zeros<dcomplex>(r, N, N);
  Tkaps   = nda::zeros<dcomplex>(n, r, N, N);
  Tmu     = nda::zeros<dcomplex>(r, N, N);
  Sigma_L = nda::zeros<dcomplex>(r, N, N);
  Sigma   = nda::zeros<dcomplex>(r, N, N);
}

void DiagramEvaluator::reset() {
  T       = 0;
  GKt     = 0;
  Tkaps   = 0;
  Tmu     = 0;
  Sigma_L = 0;
  Sigma   = 0;
}

void DiagramEvaluator::multiply_vertex_dense(Backbone &backbone, int v_ix) {

  int r    = dlr_it.size();
  int o_ix = backbone.get_vertex_orb(v_ix); // orbital index
  int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  // backbone.get_vertex_hyb_ind(v_ix) = i, where i is the # of primes on l
  // l_ix = value of l with i primes

  if (backbone.has_vertex_bar(v_ix)) {   // F has bar
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fset.F_dag_bars(o_ix, l_ix, _, _), T(t, _, _));
    } else {
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fset.F_bars_refl(o_ix, l_ix, _, _), T(t, _, _));
    }
  } else {
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fset.F_dags(o_ix, _, _), T(t, _, _));
    } else {
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fset.Fs(o_ix, _, _), T(t, _, _));
    }
  }

  // K factor
  int bv      = backbone.get_vertex_Ksign(v_ix); // sign on K
  double pole = dlr_rf(l_ix);
  if (bv != 0) {
    for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), bv * pole) * T(t, _, _); }
  }
}

void DiagramEvaluator::compose_with_edge_dense(Backbone &backbone, int e_ix) {

  GKt = Gt;
  int m         = backbone.m;
  int r         = dlr_it.size();
  for (int x = 0; x < m - 1; x++) {
    int be = backbone.get_edge(e_ix, x); // sign on K
    if (be != 0) {
      for (int t = 0; t < r; t++) {
        GKt(t, _, _) = k_it(dlr_it(t), be * dlr_rf(backbone.get_pole_ind(x))) * GKt(t, _, _);
      }
    }
  }
  T = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(T),
                                         TIME_ORDERED);
}

void DiagramEvaluator::multiply_zero_vertex(Backbone &backbone, bool is_forward) {

  int n = backbone.n;
  int r = hyb.extent(0);
  if (is_forward) {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) { Tkaps(kap, t, _, _) = nda::matmul(T(t, _, _), Fset.Fs(kap, _, _)); }
    }
    T = 0;
    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) { Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _); }
      }
      for (int t = 0; t < r; t++) { T(t, _, _) += nda::matmul(Fset.F_dags(mu, _, _), Tmu(t, _, _)); }
    }
  } else {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) { Tkaps(kap, t, _, _) = nda::matmul(T(t, _, _), Fset.F_dags(kap, _, _)); }
    }
    T = 0;
    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) { Tmu(t, _, _) += hyb_refl(t, mu, kap) * Tkaps(kap, t, _, _); }
      }
      for (int t = 0; t < r; t++) { T(t, _, _) += nda::matmul(Fset.Fs(mu, _, _), Tmu(t, _, _)); }
    }
  }
}

void DiagramEvaluator::eval_diagram_dense(Backbone &backbone) {

  int m = backbone.m;
  for (int fb = 0; fb < pow(2, m); fb++) { // loop over 2^m combos of for/backward hybridization lines
    int fb0 = fb;
    // turn (int) fb into a vector of forward/backward indices
    auto fb_vec = nda::vector<int>(m);
    for (int i = 0; i < m; i++) {
      fb_vec(i) = fb0 % 2; // 0 for backward, 1 for forward
      fb0 /= 2;
    }
    backbone.set_directions(fb_vec);
    eval_diagram_fixed_lines_dense(backbone); // evaluate the diagram with these directions
    backbone.reset_directions();
  }
}

void DiagramEvaluator::eval_diagram_fixed_lines_dense(Backbone &backbone) {
  int r = itops.rank(), m = backbone.m;
  // L = pole multiindex
  for (int L = 0; L < pow(r, m - 1); L++) { // loop over all combinations of pole indices
    Sigma_L = 0;
    int L0            = L;
    // turn (int) L into a vector of pole indices
    auto pole_inds = nda::vector<int>(m - 1);
    for (int i = 0; i < m - 1; i++) {
      pole_inds(i) = L0 % r;
      L0 /= r;
    }
    backbone.set_pole_inds(pole_inds, dlr_rf);

    eval_diagram_fixed_poles_lines_dense(backbone);
    int sign = (m % 2 == 0) ? -1 : 1;
    Sigma += sign * Sigma_L;

    backbone.reset_pole_inds();
  }
}

void DiagramEvaluator::eval_diagram_fixed_poles_lines_dense(Backbone &backbone) {
  int n = backbone.n, m = backbone.m;

  // set up orbital (Greek) indices that are explicitly summed over
  for (int s = 0; s < pow(n, m - 1); s++) { // loop over all combos of orbital indices
    int s0 = s;
    // turn (int) s into a vector of orbital indices
    auto orb_inds = nda::vector<int>(m);
    for (int i = 1; i < m; i++) { // loop over lines, skipping the one connected to vertex 0
      orb_inds(backbone.get_topology(i, 0)) = s0 % n;
      orb_inds(backbone.get_topology(i, 1)) = s0 % n;
      // orbital indices on vertices connected by a line are the same
      s0 /= n;
    }
    backbone.set_orb_inds(orb_inds);

    eval_diagram_fixed_orbs_poles_lines_dense(backbone);

    Sigma_L += T;
    backbone.reset_orb_inds(); // reset orbital indices for the next iteration
  }

  // Multiply by prefactor
  for (int p = 0; p < m - 1; p++) {           // loop over hybridization indices
    int exp = backbone.get_prefactor_Kexp(p); // exponent on K for this hybridization index
    if (exp != 0) {
      int Ksign = backbone.get_prefactor_Ksign(p);            // sign on K for this hybridization index
      double om = dlr_rf(backbone.get_pole_ind(p)); // DLR frequency for this value of this hybridization index
      double k  = k_it(0, Ksign * om);
      for (int q = 0; q < exp; q++) Sigma_L /= k;
    }
  }
  Sigma_L *= backbone.prefactor_sign;
}

void DiagramEvaluator::eval_diagram_fixed_orbs_poles_lines_dense(Backbone &backbone) {
  int m = backbone.m;

  // 1. Starting from tau_1, proceed right to left, performing multiplications at vertices and convolutions at edges, until reaching the vertex
  // containing the undecomposed hybridization line Delta_{mu kappa}.
  T = Gt; // T stores the result moving left to right
  // T is initialized to Gt, which is always the function at the rightmost edge
  for (int v = 1; v < backbone.get_topology(0, 1); v++) { // loop from the first vertex to before the special vertex
    multiply_vertex_dense(backbone, v);
    compose_with_edge_dense(backbone, v);
  }

  // 2. For each kappa, multiply by F_kappa(^dag). Then for each mu, kappa, multiply by Delta_{mu kappa}, and sum over kappa. Finally for each mu,
  // multiply F_mu[^dag] and sum over mu.
  multiply_zero_vertex(backbone, (not backbone.has_vertex_dag(0)));

  // 3. Continue right to left until the final vertex multiplication is complete.
  for (int v = backbone.get_topology(0, 1) + 1; v < 2 * m; v++) { // loop from the special vertex to the last vertex
    compose_with_edge_dense(backbone, v - 1);
    multiply_vertex_dense(backbone, v);
  }
}