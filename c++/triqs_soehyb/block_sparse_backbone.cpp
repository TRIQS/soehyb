#include <cppdlr/dlr_imtime.hpp>
#include <nda/nda.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_backbone.hpp>

using namespace nda;

BackboneVertex::BackboneVertex() : bar(false), dag(false), pole_prime(0), Ksign(0), orb(0) {;} 

bool BackboneVertex::has_bar() {return bar;}
bool BackboneVertex::has_dag() {return dag;}
int BackboneVertex::get_pole_prime() {return pole_prime;}
int BackboneVertex::get_Ksign() {return Ksign;}
int BackboneVertex::get_orb() {return orb;}

void BackboneVertex::set_bar(bool b) {bar = b;}
void BackboneVertex::set_dag(bool b) {dag = b;}
void BackboneVertex::set_pole_prime(int i) {pole_prime = i;}
void BackboneVertex::set_Ksign(int i) {Ksign = i;}
void BackboneVertex::set_orb(int i) {orb = i;}

BackboneSignature::BackboneSignature(nda::array<int,2> topology, int n) : 
    topology(topology), m(topology.extent(0)), n(n), // e.g. 2CA, m = 3, topology has 3 lines 
    prefactor_sign(1) {
    
    prefactor_Ksigns = nda::vector<int>(m-1, 0); 
    // for each of m-1 pole indices (l, l`, ...), the sign on K_l^?(0)
    prefactor_Kexps = nda::vector<int>(m-1, 0); 
    // for each of m-1 pole_indices, the exponent on K_l(0)^?
    
    auto dummy = BackboneVertex();
    vertices = std::vector<BackboneVertex>(2*m, dummy); 
    
    edges = nda::zeros<int>(2*m-1, m-1); // TODO: better name
    // 2*m-1 = # edges; m-1: each pole index
    // edges(e,p) = exponent on K_l(tau) on edge e, where l has p primes
    // TODO: flesh out comments
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
    vertices[0].set_bar(false); // operator on vertex 0 has no bar
    vertices[topology(0,1)].set_bar(false); // operator on vertex connected to 0 has no bar
    if (fb(0) == 1) {
        vertices[0].set_dag(false); // annihilation operator on vertex 0
        vertices[topology(0,1)].set_dag(true); // creation operator on vertex connected to 0
    } else {
        vertices[0].set_dag(true); // creation operator on vertex 0 
        vertices[topology(0,1)].set_dag(false); // annihilation operator on vertex connected to 0
    }

    for (int i = 1; i < m; i++) {
        vertices[topology(i,0)].set_bar(false); // operator on vertex i has no bar
        vertices[topology(i,1)].set_bar(true); // operator on vertex connected to i has a bar
        if (fb(i) == 1) {
            vertices[topology(i,0)].set_dag(false); // annihilation operator on vertex i
            vertices[topology(i,1)].set_dag(true); // creation operator on vertex i
        } else {
            vertices[topology(i,0)].set_dag(true); // creation operator on vertex i
            vertices[topology(i,1)].set_dag(false); // annihilation operator on vertex i
        }
    }
}

void BackboneSignature::reset_directions() {
    fb = 0;
    for (int i = 0; i < 2*m; i++) {
        vertices[i].set_bar(false); 
        vertices[i].set_dag(false); 
    }
}

void BackboneSignature::set_pole_inds(
    nda::vector_const_view<int> pole_inds, 
    nda::vector_const_view<double> dlr_rf) {
    // @param[in] poles DLR/AAA poles (e.g. l, l' indices)

    this->pole_inds = pole_inds;
    for (int i = 1; i < m; i++) {
        if (fb(i) == 1) { // line i is forward
            if (dlr_rf(pole_inds(i-1)) <= 0) {
                // step 4(a)
                // place K^-_l F_nu at the right vertex
                vertices[topology(i,0)].set_pole_prime(i-1); 
                vertices[topology(i,0)].set_Ksign(-1); 
                // place K^+_l F^bar^dag_nu_l at the left vertex
                vertices[topology(i,1)].set_pole_prime(i-1);
                vertices[topology(i,1)].set_Ksign(1);
                // divide by K^-_l(0)
                prefactor_Ksigns(i-1) = -1; 
                prefactor_Kexps(i-1) = 1; 
            } else {
                // step 4(b)
                // no K's on vertices, but F bar on left vertex
                vertices[topology(i,1)].set_pole_prime(i-1); 
                // place K^+_l on each edge between the two vertices
                for (int j = topology(i,0); j < topology(i,1); j++) edges(j, i-1) = 1;
                // divide by (K^+_l(0))^(# edges between vertices - 1)
                prefactor_Ksigns(i-1) = 1; 
                prefactor_Kexps(i-1) = topology(i,1) - topology(i,0) - 1; 
            }
        }
        else { // line i is backward
            if (dlr_rf(pole_inds(i-1)) >= 0) {
                // step 4(a)
                // place K^+_l F^dag_pi at the right vertex
                vertices[topology(i,0)].set_pole_prime(i-1); 
                vertices[topology(i,0)].set_Ksign(1);
                // place K^-_l F^bar_pi_l at the left vertex
                vertices[topology(i,1)].set_pole_prime(i-1);
                vertices[topology(i,1)].set_Ksign(-1);
                // divide by -K^-+l(0)
                prefactor_sign *= -1; 
                prefactor_Ksigns(i-1) = 1; 
                prefactor_Kexps(i-1) = 1; 
            } else {
                // step 4(b)
                // no K's on vertices, but F bar on left vertex
                vertices[topology(i,1)].set_pole_prime(i-1); 
                // place K^-_l on each edge between the two vertices
                for (int j = topology(i,0); j < topology(i,1); j++) edges(j, i-1) = -1;
                // divide by (-K^-_l(0))^(# edges between vertices - 1)
                prefactor_sign *= -1; 
                // if ((topology(i,1) - topology(i,0) - 1) % 2 == 1) prefactor_sign *= -1; 
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
    // vertex_which_pole_ind = 0; 
    // vertex_Ksigns = 0; 
    // vertex_states = 0; 
    for (int i = 0; i < 2*m; i++) {
        vertices[i].set_pole_prime(0);
        vertices[i].set_Ksign(0);
        vertices[i].set_orb(0); 
    }
}

void BackboneSignature::set_states(nda::vector_const_view<int> states) {
    // @param[in] states orbital+spin indices (e.g. lambda, mu indices), going 
    // right to left, excluding the ones associated with the special vertex

    for (int i = 0; i < 2*m; i++) {
        if (i != 0 && i != topology(0,1)) {
            // vertex_states(i) = states(i); 
            vertices[i].set_orb(states(i)); 
        }
    }
}

void BackboneSignature::reset_states() {
    // vertex_states = 0; 
    for (int i = 0; i < 2*m; i++) vertices[i].set_orb(0); 
}

int BackboneSignature::get_prefactor_Ksign(int i) {return prefactor_Ksigns(i);}
int BackboneSignature::get_prefactor_Kexp(int i) {return prefactor_Kexps(i);}
bool BackboneSignature::has_vertex_bar(int i) {return vertices[i].has_bar();}
bool BackboneSignature::has_vertex_dag(int i) {return vertices[i].has_dag();}
int BackboneSignature::get_vertex_pole_prime(int i) {return vertices[i].get_pole_prime();}
int BackboneSignature::get_vertex_Ksign(int i) {return vertices[i].get_Ksign();}
int BackboneSignature::get_vertex_orb(int i) {return vertices[i].get_orb();}
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
            for (int j = 0; j < B.get_vertex_pole_prime(i); j++) v_str_tmp += "`";
            v_str_tmp += "}";
            if (B.get_vertex_Ksign(i) == 1) v_str_tmp += "^+";
            else v_str_tmp += "^-"; 
        }

        // F operator
        v_str_tmp += "F";
        if (B.has_vertex_bar(i) || B.has_vertex_dag(i) ) v_str_tmp += "^{"; 
        if (B.has_vertex_bar(i)) v_str_tmp += "bar";
        if (B.has_vertex_dag(i)) v_str_tmp += "dag";
        if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "}"; 
        if (not B.has_vertex_bar(i)) v_str_tmp += "_" + std::to_string(i); 
        else {
            for (int j = 0; j < B.m; j++) if (B.get_topology(j,1) == i) v_str_tmp += "_" + std::to_string(B.get_topology(j,0));
            v_str_tmp += "l"; 
            for (int j = 0; j < B.get_vertex_pole_prime(i); j++) v_str_tmp += "`";
        }
        // hybridization
        if (i == B.get_topology(0, 1)) {
            v_str_tmp += " Delta_{"; 
            if (B.has_vertex_dag(i) == 1) v_str_tmp += "0," + std::to_string(i) + "} "; 
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
    nda::array_view<dcomplex,3> T
) {
    int r = dlr_it.size();
    int s_ix = backbone.get_vertex_orb(v_ix); // state_index
    int l_ix = backbone.get_pole_ind(backbone.get_vertex_pole_prime(v_ix)); 
    // backbone.get_vertex_pole_prime(v_ix) = i, where i is the # of primes on l
    // l_ix = value of l with i primes

    if (backbone.has_vertex_bar(v_ix)) { // F has bar
        if (backbone.has_vertex_dag(v_ix)) { // F has dagger
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fdagbars(s_ix,l_ix,_,_), T(t,_,_));
        } else {
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fbarsrefl(s_ix,l_ix,_,_), T(t,_,_)); 
        }
    } else {
        if (backbone.has_vertex_dag(v_ix)) { // F has dagger
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

void multiply_vertex_dense(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    DenseFSet& Fset,  
    int v_ix, // vertex index
    nda::array_view<dcomplex,3> T
) {
    int r = dlr_it.size();
    int s_ix = backbone.get_vertex_orb(v_ix); // state_index
    int l_ix = backbone.get_pole_ind(backbone.get_vertex_pole_prime(v_ix)); 

    if (backbone.has_vertex_bar(v_ix)) { // F has bar
        if (backbone.has_vertex_dag(v_ix)) { // F has dagger
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fset.F_dag_bars(s_ix,l_ix,_,_), T(t,_,_)); 
        } else {
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fset.F_bars_refl(s_ix,l_ix,_,_), T(t,_,_)); 
        }
    } else {
        if (backbone.has_vertex_dag(v_ix)) { // F has dagger
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fset.F_dags(s_ix,_,_), T(t,_,_));
        } else {
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fset.Fs(s_ix,_,_), T(t,_,_));
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

void compose_with_edge_dense(
    BackboneSignature& backbone, 
    imtime_ops& itops, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    double beta, 
    nda::array_const_view<dcomplex,3> Gt, 
    int e_ix, // edge index
    nda::array_view<dcomplex,3> T, 
    nda::array_view<dcomplex,3> GKt
) {
    compute_edge_dense(backbone, dlr_it, dlr_rf, Gt, e_ix, GKt); 
    T = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED); 
}

// TODO: better name
void multiply_zero_vertex(
    BackboneSignature& backbone, 
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_refl, 
    bool is_forward, 
    nda::array_const_view<dcomplex,3> Fs, 
    nda::array_const_view<dcomplex,3> F_dags, 
    nda::array_view<dcomplex,4> Tkaps, 
    nda::array_view<dcomplex,3> Tmu, 
    nda::array_view<dcomplex,3> T) {

    int n = backbone.n; 
    int r = hyb.extent(0); 
    if (is_forward) {
        for (int kap = 0; kap < n; kap++) {
            for (int t = 0; t < r; t++) {
                Tkaps(kap,t,_,_) = nda::matmul(T(t,_,_), Fs(kap,_,_));
            }
        }
        T = 0; 
        for (int mu = 0; mu < n; mu++) {
            Tmu = 0;
            for (int kap = 0; kap < n; kap++) {
                for (int t = 0; t < r; t++) {
                    Tmu(t,_,_) += hyb(t,mu,kap)*Tkaps(kap,t,_,_);
                }
            }
            for (int t = 0; t < r; t++) {
                T(t,_,_) += nda::matmul(F_dags(mu,_,_), Tmu(t,_,_));
            }
        }
    } else {
        for (int kap = 0; kap < n; kap++) {
            for (int t = 0; t < r; t++) {
                Tkaps(kap,t,_,_) = nda::matmul(T(t,_,_), F_dags(kap,_,_));
            }
        }
        T = 0; 
        for (int mu = 0; mu < n; mu++) {
            Tmu = 0;
            for (int kap = 0; kap < n; kap++) {
                for (int t = 0; t < r; t++) {
                    Tmu(t,_,_) += hyb_refl(t,mu,kap)*Tkaps(kap,t,_,_);
                }
            }
            for (int t = 0; t < r; t++) {
                T(t,_,_) += nda::matmul(Fs(mu,_,_), Tmu(t,_,_));
            }
        }
    }
}

void eval_backbone_s_p_d_dense(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags, 
    nda::array_const_view<dcomplex, 4> Fdagbars, 
    nda::array_const_view<dcomplex, 4> Fbarsrefl, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    nda::array_view<dcomplex, 3> T, 
    nda::array_view<dcomplex, 3> GKt, 
    nda::array_view<dcomplex, 4> Tkaps, 
    nda::array_view<dcomplex, 3> Tmu, 
    nda::array_view<dcomplex, 3> hyb_refl
) {
    int m = backbone.m; 
    // 1. Starting from tau_1, proceed right to left, performing 
    //    multiplications at vertices and convolutions at edges, 
    //    until reaching the vertex containing the undecomposed 
    //    hybridization line Delta_{mu kappa}
    T = Gt; // T stores the result as you move right to left
    // T is initialized with Gt, which is always the function at the rightmost edge
    for (int v = 1; v < backbone.get_topology(0,1); v++) { // loop from the first vertex to before the vertex connected to zero
        multiply_vertex_dense(backbone, dlr_it, dlr_rf, Fs, F_dags, Fdagbars, Fbarsrefl, v, T);
        compose_with_edge_dense(backbone, itops, dlr_it, dlr_rf, beta, Gt, v, T, GKt); 
    }

    // 2. For each kappa, multiply by F_kappa(^dag). Then for each 
    //    mu, kappa, multiply by Delta_{mu kappa}, and sum over 
    //    kappa. Finally for each mu, multiply F_mu[^dag] and sum 
    //    over mu. 
    /*
    if (not backbone.has_vertex_dag(0)) { // line connected to zero is forward
        multiply_zero_vertex(backbone, hyb, Fs, F_dags, Tkaps, Tmu, T);
    } else { // line connected to zero is backward
        multiply_zero_vertex(backbone, hyb_refl, F_dags, Fs, Tkaps, Tmu, T); 
    }
    */
    multiply_zero_vertex(backbone, hyb, hyb_refl, (not backbone.has_vertex_dag(0)), Fs, F_dags, Tkaps, Tmu, T);

    // 3. Continue right to left until the final vertex 
    //    multiplication is complete.
    for (int v = backbone.get_topology(0,1) + 1; v < 2*m; v++) { // loop from the next edge to the final vertex
        compose_with_edge_dense(backbone, itops, dlr_it, dlr_rf, beta, Gt, v-1, T, GKt); 
        multiply_vertex_dense(backbone, dlr_it, dlr_rf, Fs, F_dags, Fdagbars, Fbarsrefl, v, T); 
        // if (v == 5) std::cout << "middle of s p d " << T(10,_,_) << std::endl; 
    }
}

void eval_backbone_p_d_dense(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags, 
    nda::array_const_view<dcomplex, 4> Fdagbars, 
    nda::array_const_view<dcomplex, 4> Fbarsrefl, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    nda::array_view<dcomplex, 3> T, 
    nda::array_view<dcomplex, 3> GKt, 
    nda::array_view<dcomplex, 4> Tkaps, 
    nda::array_view<dcomplex, 3> Tmu, 
    nda::array_view<dcomplex, 3> hyb_refl, 
    nda::vector_view<int> states, 
    nda::array_view<dcomplex, 3> Sigma_L
) {
    int n = backbone.n, m = backbone.m; 
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
        backbone.set_states(states);
        eval_backbone_s_p_d_dense(
            backbone, beta, itops, hyb, Gt, Fs, F_dags, 
            Fdagbars, Fbarsrefl, dlr_it, dlr_rf, 
            T, GKt, Tkaps, Tmu, hyb_refl);
        Sigma_L += T; 
        backbone.reset_states(); 
    } // sum over states

    // 4. Multiply by prefactor
    for (int p = 0; p < m-1; p++) { // loop over pole indices
        int exp = backbone.get_prefactor_Kexp(p); // exponent on K for this pole index
        if (exp != 0) { 
            int Ksign = backbone.get_prefactor_Ksign(p); 
            double om = dlr_rf(backbone.get_pole_ind(p)); 
            double k = k_it(0, Ksign*om); 
            for (int q = 0; q < exp; q++) Sigma_L = Sigma_L / k; 
        }
    }
    Sigma_L = backbone.prefactor_sign * Sigma_L; 
}

void eval_backbone_d_dense(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags, 
    nda::array_const_view<dcomplex, 4> Fdagbars, 
    nda::array_const_view<dcomplex, 4> Fbarsrefl, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    nda::array_view<dcomplex, 3> T, 
    nda::array_view<dcomplex, 3> GKt, 
    nda::array_view<dcomplex, 4> Tkaps, 
    nda::array_view<dcomplex, 3> Tmu, 
    nda::array_view<dcomplex, 3> hyb_refl, 
    nda::vector_view<int> states, 
    nda::array_view<dcomplex, 3> Sigma_L, 
    nda::vector_view<int> pole_inds, 
    int sign, 
    nda::array_view<dcomplex, 3> Sigma
) {
    int r = itops.rank(), m = backbone.m; 
    // L = pole multiindex
    for (int L = 0; L < pow(r,m-1); L++) { // loop over all combinations of pole indices
        Sigma_L = 0; 

        int L0 = L;
        // turn (int) L into a vector of pole indices
        for (int i = 0; i < m-1; i++) {pole_inds(i) = L0 % r; L0 = L0 / r;}
        backbone.set_pole_inds(pole_inds, dlr_rf); // give pole indices to backbone object

        // print a negative and positive pole diagram for each set of line directions
        if (L == 0 || L == pow(r,m-1)/4 || L == pow(r,m-1)/2 || L == 3*pow(r,m-1)/4) {
            std::cout << "poles = ";
            for (int i = 0; i < m-1; i++) {
                std::cout << dlr_rf(pole_inds(i)) << ", ";
            }
            std::cout << backbone << std::endl;
        }
        eval_backbone_p_d_dense(
            backbone, beta, itops, hyb, Gt, Fs, F_dags, 
            Fdagbars, Fbarsrefl, dlr_it, dlr_rf, 
            T, GKt, Tkaps, Tmu, hyb_refl, states, Sigma_L); 
        Sigma += sign*Sigma_L; 
        // if (backbone.m == 3 && fb == 7) Sigma_temp += sign*backbone.prefactor_sign*Sigma_L; 
        backbone.reset_pole_inds(); 
    }
}

nda::array<dcomplex, 3> eval_backbone_dense(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags
) {
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
    auto hyb_refl = itops.reflect(hyb);  // nda::make_regular(-itops.reflect(hyb));
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
    nda::array<dcomplex,3> Sigma(r,N,N), Sigma_temp(r,N,N);

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
        int sign = (fb==0 || fb==1 || fb==6 || fb==7) ? 1 : -1; // ((fb + m) % 2 == 0) ? 1 : -1; // (fb_vec(0)^fb_vec(1)) ? -1 : 1; // TODO: figure this out
        std::cout << "\nDiagrams, fb = " << fb << std::endl;
        eval_backbone_d_dense(
            backbone, beta, itops, hyb, Gt, Fs, F_dags, Fdagbars, Fbarsrefl, 
            dlr_it, dlr_rf, T, GKt, Tkaps, Tmu, hyb_refl, states, Sigma_L, 
            pole_inds, sign, Sigma); 
        backbone.reset_directions(); 
    } // sum over forward/backward lines

    return Sigma;
}

nda::array<dcomplex, 3> eval_backbone_dense(BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    DenseFSet& Fset
) {

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
    int n = hyb.extent(1);
    auto hyb_refl = itops.reflect(hyb); // nda::make_regular(-itops.reflect(hyb));

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
        int sign = (fb==0 || fb==2) ? 1 : -1; // (fb_vec(0)^fb_vec(1)) ? -1 : 1; // TODO: figure this out

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
                backbone.set_states(states); 

                // 1. Starting from tau_1, proceed right to left, performing 
                //    multiplications at vertices and convolutions at edges, 
                //    until reaching the vertex containing the undecomposed 
                //    hybridization line Delta_{mu kappa}
                T = Gt; // T stores the result as a move right to left
                // T is initialized with Gt, which is always the function at the rightmost edge
                for (int v = 1; v < backbone.get_topology(0,1); v++) { // loop from the first vertex to before the special vertex
                    // TODO: struct/class for Fs
                    multiply_vertex_dense(backbone, dlr_it, dlr_rf, Fset, v, T);
                    compose_with_edge_dense(backbone, itops, dlr_it, dlr_rf, beta, Gt, v, T, GKt); 
                } 

                // 2. For each kappa, multiply by F_kappa(^dag). Then for each 
                //    mu, kappa, multiply by Delta_{mu kappa}, and sum over 
                //    kappa. Finally for each mu, multiply F_mu[^dag] and sum 
                //    over mu.
                /*
                if (not backbone.has_vertex_dag(0)) { // line connected to zero is forward
                    multiply_zero_vertex(backbone, hyb, Fset.Fs, Fset.F_dags, Tkaps, Tmu, T); 
                } else { // line connected to zero is backward
                    multiply_zero_vertex(backbone, hyb_refl, Fset.F_dags, Fset.Fs, Tkaps, Tmu, T); 
                }
                */ 
                multiply_zero_vertex(backbone, hyb, hyb_refl, (not backbone.has_vertex_dag(0)), Fset.Fs, Fset.F_dags, Tkaps, Tmu, T);

                // 3. Continue right to left until the final vertex 
                //    multiplication is complete.
                for (int v = backbone.get_topology(0,1) + 1; v < 2*m; v++) { // loop from the next edge to the final vertex
                    compose_with_edge_dense(backbone, itops, dlr_it, dlr_rf, beta, Gt, v-1, T, GKt); 
                    multiply_vertex_dense(backbone, dlr_it, dlr_rf, Fset, v, T); 
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

void multiply_vertex_block(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    std::vector<BlockOp>& Fs, 
    std::vector<BlockOp>& F_dags, 
    std::vector<std::vector<BlockOp>>& Fdagbars, 
    std::vector<std::vector<BlockOp>>& Fbarsrefl, 
    int v_ix, 
    int b_ix, 
    nda::array_view<dcomplex,3> T, 
    nda::vector_const_view<int> block_dims
) {
    int r = dlr_it.size();
    int s_ix = backbone.get_vertex_orb(v_ix); // state_index
    int l_ix = backbone.get_pole_ind(backbone.get_vertex_pole_prime(v_ix)); 

    if (backbone.has_vertex_bar(v_ix)) { // F has bar
        if (backbone.has_vertex_dag(v_ix)) { // F has dagger
            for (int t = 0; t < r; t++) {
                T(t,range(0,block_dims(1)),range(0,block_dims(0))) = 
                    nda::matmul(
                        Fdagbars[s_ix][l_ix].get_block(b_ix), 
                        T(t,range(0,block_dims(0)),range(0,block_dims(0))));
            } 
        } else {
            for (int t = 0; t < r; t++) {
                T(t,range(0,block_dims(1)),range(0,block_dims(0))) = 
                    nda::matmul(
                        Fbarsrefl[s_ix][l_ix].get_block(b_ix), 
                        T(t,range(0,block_dims(0)),range(0,block_dims(0))));
            }
        }
    } else {
        if (backbone.has_vertex_dag(v_ix)) { // F has dagger
            for (int t = 0; t < r; t++) {
                T(t,range(0,block_dims(1)),range(0,block_dims(0))) = 
                    nda::matmul(
                        F_dags[s_ix].get_block(b_ix), 
                        T(t,range(0,block_dims(0)),range(0,block_dims(0))));
            }
        } else {
            for (int t = 0; t < r; t++) {
                T(t,range(0,block_dims(1)),range(0,block_dims(0))) = 
                    nda::matmul(
                        Fs[s_ix].get_block(b_ix), 
                        T(t,range(0,block_dims(0)),range(0,block_dims(0))));
            }
        }
    }

    // K factor
    int bv = backbone.get_vertex_Ksign(v_ix); // sign on K
    double pole = dlr_rf(l_ix); 
    if (bv != 0) {
        for (int t = 0; t < r; t++) {
            T(t,range(0,block_dims(1)),range(0,block_dims(0))) = k_it(dlr_it(t), bv * pole) * T(t,range(0,block_dims(1)),range(0,block_dims(0)));
        }
    }
}

void compute_edge_block(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    BlockDiagOpFun& Gt, 
    int e_ix, 
    int b_ix, 
    nda::array_view<dcomplex,3> GKt_b
) {
    GKt_b = Gt.get_block(b_ix); 
    int m = backbone.m; 
    int r = dlr_it.size(); 
    for (int x = 0; x < m-1; x++) {
        int be = backbone.get_edge(e_ix, x); // sign on K
        if (be != 0) {
            for (int t = 0; t < r; t++) {
                GKt_b(t,_,_) = k_it(dlr_it(t), be * dlr_rf(backbone.get_pole_ind(x))) * GKt_b(t,_,_);
            }
        }
    }
}

void compose_with_edge_block(
    BackboneSignature& backbone, 
    imtime_ops& itops, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    double beta, 
    BlockDiagOpFun& Gt, 
    int e_ix, 
    int b_ix, 
    nda::array_view<dcomplex,3> T, 
    nda::array_view<dcomplex,3> GKt_b
) {
    compute_edge_block(backbone, dlr_it, dlr_rf, Gt, e_ix, b_ix, GKt_b); 
    T = itops.convolve(beta, Fermion, itops.vals2coefs(GKt_b), itops.vals2coefs(T), TIME_ORDERED);    
}

void multiply_zero_vertex(
    BackboneSignature& backbone, 
    nda::array_const_view<dcomplex,3> hyb, 
    std::vector<BlockOp>& F0s, 
    std::vector<BlockOp>& Fhybs, 
    nda::array_view<dcomplex,4> Tkaps, 
    nda::array_view<dcomplex,3> Tmu, 
    int b_ix, 
    nda::array_view<dcomplex,3> T) {

    int n = backbone.n; 
    int r = hyb.extent(0); 
    for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) {
            Tkaps(kap,t,_,_) = nda::matmul(T(t,_,_), F0s[kap].get_block(b_ix));
        }
    }
    T = 0; 
    for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
            for (int t = 0; t < r; t++) {
                Tmu(t,_,_) += hyb(t,mu,kap)*Tkaps(kap,t,_,_);
            }
        }
        for (int t = 0; t < r; t++) {
            T(t,_,_) += nda::matmul(Fhybs[mu].get_block(b_ix), Tmu(t,_,_)); 
        }
    }
}

BlockDiagOpFun eval_backbone(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    BlockDiagOpFun& Gt, 
    std::vector<BlockOp>& Fs, 
    std::vector<BlockOp>& F_dags
) {
    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.extent(0);
    int m = backbone.m;

    auto hyb_coeffs = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl = nda::make_regular(-itops.reflect(hyb));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
    int n = Fs[0].get_block(0).extent(0);

    // compute Fbars and Fdagbars
    auto Fbar_indices = Fs[0].get_block_indices();
    auto Fbar_sizes = Fs[0].get_block_sizes();
    auto Fdagbar_indices = F_dags[0].get_block_indices();
    auto Fdagbar_sizes = F_dags[0].get_block_sizes();
    std::vector<std::vector<BlockOp>> Fdagbars(
        n, 
        std::vector<BlockOp>(
            r, 
            BlockOp(Fdagbar_indices, Fdagbar_sizes)));
    std::vector<std::vector<BlockOp>> Fbarsrefl(
        n, 
        std::vector<BlockOp>(
            r, 
            BlockOp(Fbar_indices, Fbar_sizes)));
    for (int lam = 0; lam < n; lam++) {
        for (int l = 0; l < r; l++) {
            for (int nu = 0; nu < n; nu++) {
                Fdagbars[lam][l] += hyb_coeffs(l,nu,lam)*F_dags[nu];
                Fbarsrefl[lam][l] += hyb_refl_coeffs(l,lam,nu)*Fs[nu];
            }
        }
    }

    // initialize self-energy
    BlockDiagOpFun Sigma = BlockDiagOpFun(r, Gt.get_block_sizes());
    int bc = Gt.get_num_block_cols();

    // preallocate intermediate arrays
    // TODO: for b-s version, initialize largest needed array and write into top-left corner
    bool path_all_nonzero = true; 
    nda::vector<int> ind_path(2*m-1); 
    nda::vector<int> block_dims(2*m+1); 
    int Nmax = Gt.get_max_block_size();
    nda::array<dcomplex, 3> Sigma_L(r, Nmax, Nmax), T(r, Nmax, Nmax), Tmu(r, Nmax, Nmax), GKt(r, Nmax, Nmax); 
    nda::array<dcomplex, 4> Tkaps(n, r, Nmax, Nmax);
    // Sigma_l = term of self-energy assoc'd with pole l, rest are placeholders

    // loop over hybridization lines
    nda::vector<int> fb_vec(m), states(2*m);
    auto pole_inds = nda::zeros<int>(m-1);

    for (int fb = 0; fb < pow(2,m); fb++) { // loop over 2^m combos of for/backward lines
        int fb0 = fb;
        // turn (int) fb into a vector of 1s and 0s corresp. to forward, backward lines, resp. 
        for (int i = 0; i < m; i++) {fb_vec(i) = fb0 % 2; fb0 = fb0 / 2;}
        backbone.set_directions(fb_vec); // give line directions to backbone object
        int sign = (fb_vec(0)^fb_vec(1)) ? -1 : 1; // TODO: figure this out

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
            block_dims(0) = (backbone.has_vertex_dag(0)) ? F_dags[0].get_block_size(b,1) : Fs[0].get_block_size(b,1); 
            // block_dims(1) = (backbone.vertex_dags(0)) ? F_dags[0].get_block_size(b,0) : Fs[0].get_block_size(b,0); 
            while (w < 2*m && path_all_nonzero) { // loop over vertices
                ip = (backbone.has_vertex_dag(w)) ? F_dags[0].get_block_index(ip) : Fs[0].get_block_index(ip); 
                if (ip == -1 || (w < 2*m-1 && Gt.get_zero_block_index(ip) == -1)) {
                    path_all_nonzero = false; 
                } else {
                    if (w < 2*m-1) ind_path(w) = ip;
                    block_dims(w+1) = (backbone.has_vertex_dag(w)) ? F_dags[0].get_block_size(ip,0): Fs[0].get_block_size(ip,0); 
                }
                w += 1; 
            }

            if (path_all_nonzero) {
                auto Sigma_b = nda::make_regular(0 * Sigma.get_block(b)); 
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
                        backbone.set_states(states); 

                        // 1. Starting from tau_1, proceed right to left, performing 
                        //    multiplications at vertices and convolutions at edges, 
                        //    until reaching the vertex containing the undecomposed 
                        //    hybridization line Delta_{mu kappa}
                        T = 0; 
                        T(_,range(0,block_dims(0)),range(0,block_dims(1))) = Gt.get_block(b); // T stores the result moving right to left
                        // T is initialized with block b of Gt, which is always the function at the rightmost edge

                        // !!!!!!!!!!!!!!!!
                        // TODO: block_dims
                        // !!!!!!!!!!!!!!!!

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
                    Sigma_b += sign*backbone.prefactor_sign*Sigma_L; // TODO: block_sizes for Sigma_L too 
                    backbone.reset_pole_inds(); 
                } // sum over poles
                Sigma.add_block(b, Sigma_b); 
            } // end if(path_all_nonzero)
        } // loop over blocks
        backbone.reset_directions();
    } // sum over forward/backward lines

    return Sigma;
}