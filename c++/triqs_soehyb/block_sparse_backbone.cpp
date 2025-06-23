#include <nda/nda.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_backbone.hpp>

using namespace nda;

BackboneSignature::BackboneSignature(nda::array<int,2> topology, int n) : 
    topology(topology), m(topology.extent(0)), n(n), // e.g. 2CA, m = 3, topology has 3 lines 
    prefactor_sign(1) {
    
    prefactor_Ksigns = nda::vector<int>(m-1, 0); 
    prefactor_Kexps = nda::vector<int>(m-1, 0); 
    /*        |  sign  |  sign on K  | exponent on K |
     p   ---------------------------------------------
     o   l    |        |             |               |
     l   ---------------------------------------------
     e   l'   |        |             |               |
        ---------------------------------------------
     i   l''  |        |             |               |
     n   ---------------------------------------------
     d   ...  |        |             |               |
     e
     x
    */
    vertex_bars = nda::vector<bool>(2*m, false); 
    vertex_dags = nda::vector<bool>(2*m, false);
    vertex_which_pole_ind = nda::vector<int>(2*m, 0); 
    vertex_Ksigns = nda::vector<int>(2*m, 0); 
    vertex_states = nda::vector<int>(2*m, 0); 
    /*         | bar? | dagger? |    pole index      |  sign on K  |  state index
         --------------------------------------------------------
         0     |      |         |                    |
     v   --------------------------------------------------------
     e   1     |      |         |                    |
     r   --------------------------------------------------------
     t   2     |      |         |                    |
     e   --------------------------------------------------------
     x   ...   |      |         |                    |
     #   --------------------------------------------------------
         2*m-1 |      |         |                    |
         --------------------------------------------------------
    */
    edges = nda::zeros<int>(2*m-1, m-1); // 2*m-1 = # edges; m-1: each pole index
    /*            pole index      
              | l | l' | l'' | ... 
         -------------------------
         0    |   |    |     |    
     e   -------------------------
     d   1    |   |    |     |    
     g   -------------------------
     e   2    |   |    |     |    
     #   -------------------------
         ...  |   |    |     |    
         -------------------------
         2*m-2|   |    |     |    
         -------------------------
    */
}

void BackboneSignature::set_directions(nda::vector_const_view<int> fb) {
    // @param[in] fb forward/backward line information

    this->fb = fb; 
    if (m != fb.size()) {
        throw std::invalid_argument("topology and fb must have same # of vertices");
    }
    for (int i = 0; i < m; i++) {
        if (topology(i,0) >= topology(i,1)) {
            throw std::invalid_argument("first row of topology must contain smaller-numbered vertices");
        }
    }
    if (topology(0,0) != 0) throw std::invalid_argument("topology(0,0) must be 0"); 

    // set operator flags for each vertex, depending on fb
    vertex_bars(0) = false; // operator on vertex 0 has no bar
    vertex_bars(topology(0,1)) = false; // operator on vertex connected to 0 has no bar
    if (fb(0) == 1) {
        vertex_dags(0) = false; // annihilation operator on vertex 0
        vertex_dags(topology(0,1)) = true; // creation operator on vertex connected to 0
    } else {
        vertex_dags(0) = true; // creation operator on vertex 0
        vertex_dags(topology(0,1)) = false; // annihilation operator on vertex connected to 0
    }

    for (int i = 1; i < m; i++) {
        vertex_bars(topology(i,0)) = false; // operator on vertex i has no bar
        vertex_bars(topology(i,1)) = true; // operator on vertex connected to i has a bar
        if (fb(i) == 1) {
            vertex_dags(topology(i,0)) = false; // annihilation operator on vertex i
            vertex_dags(topology(i,1)) = true; // creation operator on vertex i
        } else {
            vertex_dags(topology(i,0)) = true; // creation operator on vertex i
            vertex_dags(topology(i,1)) = false; // annihilation operator on vertex i
        }
    }
}

void BackboneSignature::reset_directions() {
    fb = 0; 
    vertex_bars = false;
    vertex_dags = false; 
}

void BackboneSignature::set_pole_inds(
    nda::vector_const_view<int> pole_inds, 
    nda::vector_const_view<double> dlr_rf) {
    // @param[in] poles DLR/AAA poles (e.g. l, l' indices)

    this->pole_inds = pole_inds;
    for (int i = 1; i < m; i++) {
        if (fb(i) == 1) {
            if (dlr_rf(pole_inds(i-1)) <= 0) {
                // step 4(a)
                // place K^-_l F_nu at the right vertex
                vertex_which_pole_ind(topology(i,0)) = i-1; 
                vertex_Ksigns(topology(i,0)) = -1; 
                // place K^+_l F^bar^dag_nu_l at the left vertex
                vertex_which_pole_ind(topology(i,1)) = i-1;
                vertex_Ksigns(topology(i,1)) = 1;
                // divide by K^-_l(0)
                prefactor_Ksigns(i-1) = -1; 
                prefactor_Kexps(i-1) = 1; 
            } else {
                // step 4(b)
                // no K's on vertices
                // place K^+_l on each edge between the two vertices
                for (int j = topology(i,0); j < topology(i,1); j++) edges(j, i-1) = 1;
                // divide by (K^+_l(0))^(# edges between vertices - 1)
                prefactor_Ksigns(i-1) = 1; 
                prefactor_Kexps(i-1) = topology(i,1) - topology(i,0) - 1; 
            }
        }
        else {
            if (dlr_rf(pole_inds(i-1)) >= 0) {
                // step 4(a)
                // place K^+_l F^dag_pi at the right vertex
                vertex_which_pole_ind(topology(i,0)) = i-1; 
                vertex_Ksigns(topology(i,0)) = 1;
                // place K^-_l F^bar_pi_l at the left vertex
                vertex_which_pole_ind(topology(i,1)) = i-1;
                vertex_Ksigns(topology(i,1)) = -1;
                // divide by -K^-+l(0)
                prefactor_sign *= -1; 
                prefactor_Ksigns(i-1) = 1; 
                prefactor_Kexps(i-1) = 1; 
            } else {
                // step 4(b)
                // no K's on vertices
                // place K^-_l on each edge between the two vertices
                for (int j = topology(i,0); j < topology(i,1); j++) edges(j, i-1) = -1;
                // divide by -(K^-_l(0))^(# edges between vertices - 1)
                prefactor_sign *= -1; 
                prefactor_Ksigns(i-1) = -1; 
                prefactor_Kexps(i-1) = topology(i,1) - topology(i,0) - 1; 
            }
        }
    }
}

void BackboneSignature::reset_pole_inds() {
    pole_inds = 0; 
    edges = 0; 
    prefactor_sign = 1; 
    prefactor_Ksigns = 0; 
    prefactor_Kexps = 0; 
    vertex_which_pole_ind = 0; 
    vertex_Ksigns = 0; 
    vertex_states = 0; 
}

void BackboneSignature::set_states(nda::vector_const_view<int> states) {
    // @param[in] states orbital+spin indices (e.g. lambda, mu indices), going 
    // right to left, excluding the ones associated with the special vertex

    for (int i = 0; i < 2*m; i++) {
        if (i != 0 && i != topology(0,1)) {
            // vertices(i,4) = states(i); 
            vertex_states(i) = states(i); 
        }
    }
}

void BackboneSignature::reset_states() {
    vertex_states = 0; 
}

int BackboneSignature::get_prefactor_Ksign(int i) {return prefactor_Ksigns(i);}

int BackboneSignature::get_prefactor_Kexp(int i) {return prefactor_Kexps(i);}

bool BackboneSignature::get_vertex_bar(int i) {return vertex_bars(i);}

bool BackboneSignature::get_vertex_dag(int i) {return vertex_dags(i);}

int BackboneSignature::get_vertex_which_pole_ind(int i) {return vertex_which_pole_ind(i);}

int BackboneSignature::get_vertex_Ksign(int i) {return vertex_Ksigns(i);}

int BackboneSignature::get_vertex_state(int i) {return vertex_states(i);}

int BackboneSignature::get_edge(int num, int pole_ind) {return edges(num, pole_ind);}

int BackboneSignature::get_topology(int i, int j) {return topology(i, j);}

int BackboneSignature::get_pole_ind(int i) {return pole_inds(i);}

std::ostream& operator<<(std::ostream& os, BackboneSignature &B) {
    // prefactor --> p_str
    std::string p_str = "1 / ("; 
    int sign = B.prefactor_sign; 
    if (sign == -1) p_str = "-" + p_str; 
    for (int i = 0; i < B.m-1; i++) {
        if (B.get_prefactor_Kexp(i) >= 1) {
            p_str += "K_{l";
            for (int j = 0; j < i; j++) p_str += "`";
            p_str += "}";
            if (B.get_prefactor_Ksign(i) == 1) p_str += "^+";
            else p_str += "^-"; 
        }
        if (B.get_prefactor_Kexp(i) > 1) {
            p_str += "^" + std::to_string(B.get_prefactor_Kexp(i)); 
        }
        p_str += "(0)";
        if (i < B.m-2) p_str += " ";
    }
    p_str += ")";

    int diag_str_cent = 30; 
    std::string diag_str = ""; 
    for (int i = 0; i < diag_str_cent; i++) diag_str += " "; 
    diag_str += "0\n"; 
    std::string v_str_tmp = "", e_str_tmp = "";
    for (int i = 0; i < 2*B.m; i++) {
        // K factor
        if (B.get_vertex_Ksign(i) != 0) {
            v_str_tmp += "K_{l";
            for (int j = 0; j < B.get_vertex_which_pole_ind(i); j++) v_str_tmp += "`";
            v_str_tmp += "}";
            if (B.get_vertex_Ksign(i) == 1) v_str_tmp += "^+";
            else v_str_tmp += "^-"; 
        }

        // F operator
        v_str_tmp += "F";
        if (B.get_vertex_bar(i) || B.get_vertex_dag(i) ) v_str_tmp += "^{"; 
        if (B.get_vertex_bar(i)) v_str_tmp += "bar";
        if (B.get_vertex_dag(i)) v_str_tmp += "dag";
        if (B.get_vertex_bar(i) || B.get_vertex_dag(i)) v_str_tmp += "}"; 
        if (not B.get_vertex_bar(i)) v_str_tmp += "_" + std::to_string(i); 
        else {
            for (int j = 0; j < B.m; j++) if (B.get_topology(j,1) == i) v_str_tmp += "_" + std::to_string(B.get_topology(j,0));
        }
        // hybridization
        if (i == B.get_topology(0, 1)) {
            v_str_tmp += " Delta_{"; 
            if (B.get_vertex_dag(i) == 1) v_str_tmp += "0," + std::to_string(i) + "} "; 
            else v_str_tmp += std::to_string(i) + ",0}";
        } else v_str_tmp += " ";
        int vlen0 = v_str_tmp.size(); 
        for (int j = vlen0; j < diag_str_cent-2; j++) v_str_tmp = " " + v_str_tmp; 
        diag_str += v_str_tmp + "--| \n"; 
        v_str_tmp = ""; 

        // edges --> e_str
        if (i < 2*B.m-1) {
            for (int j = 0; j < B.m-1; j++) {
                if (B.get_edge(i, j) != 0) {
                    e_str_tmp += "K_{l";
                    for (int k = 0; k < j; k++) e_str_tmp += "`";
                    e_str_tmp += "}";
                    if (B.get_edge(i, j) == 1) e_str_tmp += "^+ "; 
                    else e_str_tmp += "^- "; 
                }
            }
            e_str_tmp += "G ";
            for (int j = 0; j < diag_str_cent; j++) diag_str += " ";
            diag_str += "| " + e_str_tmp + "\n"; 
            e_str_tmp = ""; 
        }
    }
    // diag_str += "\"; 
    for (int i = 0; i < diag_str_cent-1; i++) diag_str += " "; 
    diag_str += "tau"; 

    os << "\nPrefactor: " << p_str << "\nDiagram: \n" << diag_str;
    return os;
}

void multiply_vertex_dense(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    nda::array_const_view<dcomplex,3> Fs, 
    nda::array_const_view<dcomplex,3> F_dags, 
    nda::array_const_view<dcomplex,4> Fdagbars, 
    nda::array_const_view<dcomplex,4> Fbarsrefl, 
    int v_ix, // vertex index
    nda::array_view<dcomplex,3> T) {

    int r = dlr_it.size();
    int s_ix = backbone.get_vertex_state(v_ix); // state_index
    int l_ix = backbone.get_pole_ind(backbone.get_vertex_which_pole_ind(v_ix)); 

    if (backbone.get_vertex_bar(v_ix)) { // F has bar
        if (backbone.get_vertex_dag(v_ix)) { // F has dagger
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fdagbars(s_ix,l_ix,_,_), T(t,_,_)); 
        } else {
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fbarsrefl(s_ix,l_ix,_,_), T(t,_,_)); 
        }
    } else {
        if (backbone.get_vertex_dag(v_ix)) { // F has dagger
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(F_dags(s_ix,_,_), T(t,_,_));
        } else {
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fs(s_ix,_,_), T(t,_,_));
        }
    }

    // K factor
    int bv = backbone.get_vertex_Ksign(v_ix); // sign on K
    double pole = dlr_rf(l_ix); 
    if (bv != 0) {
        for (int t = 0; t < r; t++) {
            T(t,_,_) = k_it(dlr_it(t), bv * pole) * T(t,_,_);
        }
    }
}

void compute_edge_dense(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    nda::array_const_view<dcomplex,3> Gt, 
    int e_ix, // edge index
    nda::array_view<dcomplex,3> GKt) {

    GKt = Gt; 
    int m = backbone.m; 
    int r = dlr_it.size(); 
    for (int x = 0; x < m-1; x++) {
        int be = backbone.get_edge(e_ix, x); // sign on K
        if (be != 0) {
            for (int t = 0; t < r; t++) {
                GKt(t,_,_) = k_it(dlr_it(t), be * dlr_rf(backbone.get_pole_ind(x))) * GKt(t,_,_);
            }
        }
    }
}

void hyb_vertex(
    BackboneSignature& backbone, 
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> F0s, 
    nda::array_const_view<dcomplex,3> Fhybs, 
    nda::array_view<dcomplex,4> Tkaps, 
    nda::array_view<dcomplex,3> Tmu, 
    nda::array_view<dcomplex,3> T) {

    int n = backbone.n; 
    int r = hyb.extent(0); 
    for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) {
            Tkaps(kap,t,_,_) = nda::matmul(T(t,_,_), F0s(kap,_,_));
        }
    }
    for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
            for (int t = 0; t < r; t++) {
                Tmu(t,_,_) += hyb(t,mu,kap)*Tkaps(kap,t,_,_);
            }
        }
        for (int t = 0; t < r; t++) {
            T(t,_,_) += nda::matmul(Fhybs(mu,_,_), Tmu(t,_,_)); 
        }
    }
}

nda::array<dcomplex, 3> eval_backbone_dense(BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags) {

    // index orders:
    // Gt (time, N, N), where N = 2^n, n = number of orbital indices
    // Fs (n, N, N)
    // Fbars (n, r, N, N)

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.extent(0);
    int N = Gt.extent(1);
    int m = backbone.m;

    auto hyb_coeffs = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl = nda::make_regular(-itops.reflect(hyb));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
    int n = Fs.extent(0);

    // compute Fbars and Fdagbars
    auto Fdagbars = nda::array<dcomplex, 4>(n, r, N, N);
    auto Fbarsrefl = nda::array<dcomplex, 4>(n, r, N, N);
    for (int lam = 0; lam < n; lam++) {
        for (int l = 0; l < r; l++) {
            for (int nu = 0; nu < n; nu++) {
                Fdagbars(lam,l,_,_) += hyb_coeffs(l,nu,lam)*F_dags(nu,_,_);
                Fbarsrefl(nu,l,_,_) += hyb_refl_coeffs(l,nu,lam)*Fs(lam,_,_);
            }
        }
    }

    // initialize self-energy
    nda::array<dcomplex,3> Sigma(r,N,N);

    // preallocate intermediate arrays
    nda::array<dcomplex, 3> Sigma_L(r, N, N), T(r, N, N), Tmu(r, N, N), GKt(r, N, N); // TODO: T --> temp_result? for b-s version, initialize largest 
    // needed array and write into top-left corner
    nda::array<dcomplex, 4> Tkaps(n, r, N, N);
    // Sigma_l = term of self-energy assoc'd with pole l, rest are placeholders
    // loop over hybridization lines

    nda::vector<int> fb_vec(m), states(2*m);
    auto pole_inds = nda::zeros<int>(m-1);

    for (int fb = 0; fb < pow(2,m); fb++) { // loop over 2^m combos of for/backward lines
        int fb0 = fb;
        // turn (int) fb into a vector of 1s and 0s corresp. to forward, backward lines, resp. 
        for (int i = 0; i < m; i++) {fb_vec(i) = fb0 % 2; fb0 = fb0 / 2;}
        backbone.set_directions(fb_vec); // give line directions to backbone object
        int sign = (fb==2 || fb==0) ? 1 : -1;  // (fb_vec(0)^fb_vec(1)) ? -1 : 1; // TODO: figure this out

        // L = pole multiindex
        for (int L = 0; L < pow(r,m-1); L++) { // loop over all combinations of pole indices
            Sigma_L = 0; 

            int L0 = L;
            // turn (int) L into a vector of pole indices
            for (int i = 0; i < m-1; i++) {pole_inds(i) = L0 % r; L0 = L0 / r;}
            backbone.set_pole_inds(pole_inds, dlr_rf); // give pole indices to backbone object

            // print a negative and positive pole diagram for each set of line directions
            if (L == 0) std::cout << "fb = " << fb << " " << backbone << std::endl;
            if (L == r-1) std::cout << "fb = " << fb << " " << backbone << std::endl;

            // set up state (Greek) indices that are explicity summed over
            for (int s = 0; s < pow(n, m-1); s++) { // loop over all combos of states
                int s0 = s;
                // turn (int) s into a vector of state indices
                for (int i = 1; i < m; i++) { // loop over lines, skipping the one connected to zero
                    states(backbone.get_topology(i, 0)) = s0 % n;
                    states(backbone.get_topology(i, 1)) = s0 % n;
                    // state indices on vertices connected by a line are the same
                    s0 = s0 / n; 
                }
                // TODO: set_states routine, remove states argument from multiply_vertex_dense, etc. 
                backbone.set_states(states); 

                // 1. Starting from tau_1, proceed right to left, performing 
                //    multiplications at vertices and convolutions at edges, 
                //    until reaching the vertex containing the undecomposed 
                //    hybridization line Delta_{mu kappa}
                T = Gt; // T stores the result as a move right to left
                // T is initialized with Gt, which is always the function at the rightmost edge
                for (int v = 1; v < backbone.get_topology(0,1); v++) { // loop from the first vertex to before the special vertex
                    // compute vertex (multiply)
                    // TODO: struct/class for Fs
                    multiply_vertex_dense(backbone, dlr_it, dlr_rf, Fs, F_dags, Fdagbars, Fbarsrefl, v, T); 
                    compute_edge_dense(backbone, dlr_it, dlr_rf, Gt, v, GKt); 
                    // convolve with edge
                    T = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED); 
                    // TODO: combine compute_edge... and convolve into one routine called compose_with_edge...
                } 

                // 2. For each kappa, multiply by F_kappa(^dag). Then for each 
                //    mu, kappa, multiply by Delta_{mu kappa}, and sum over 
                //    kappa. Finally for each mu, multiply F_mu[^dag] and sum 
                //    over mu. 
                if (not backbone.get_vertex_dag(0)) { // line connected to zero is forward
                    hyb_vertex(backbone, hyb, Fs, F_dags, Tkaps, Tmu, T); 
                } else { // line connected to zero is backward
                    hyb_vertex(backbone, hyb_refl, F_dags, Fs, Tkaps, Tmu, T); 
                }

                // 3. Continue right to left until the final vertex 
                //    multiplication is complete.
                for (int v = backbone.get_topology(0,1) + 1; v < 2*m; v++) { // loop from the next edge to the final vertex
                    compute_edge_dense(backbone, dlr_it, dlr_rf, Gt, v-1, GKt);
                    T = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED); 
                    multiply_vertex_dense(backbone, dlr_it, dlr_rf, Fs, F_dags, Fdagbars, Fbarsrefl, v, T); 
                }
                Sigma_L += T; 
                backbone.reset_states(); 
            } // sum over states

            // 4. Multiply by prefactor
            for (int p = 0; p < m-1; p++) { // loop over pole indices
                int exp = backbone.get_prefactor_Kexp(p); // exponent on K for this pole index
                if (exp != 0) { 
                    int Ksign = backbone.get_prefactor_Ksign(p); 
                    double om = dlr_rf(pole_inds(p)); 
                    double k = k_it(0, Ksign*om); 
                    for (int q = 0; q < exp; q++) Sigma_L = Sigma_L / k; 
                }
            }
            Sigma += sign*backbone.prefactor_sign*Sigma_L; 
            backbone.reset_pole_inds(); 
        } // sum over poles
        backbone.reset_directions(); 
    } // sum over forward/backward lines

    return Sigma;
}
