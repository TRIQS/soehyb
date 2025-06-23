#include <nda/declarations.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <nda/nda.hpp>

using namespace nda; 

/**
 * @class BackboneSignature
 * @brief Identifier of a diagram with specific signs on poles
 */
class BackboneSignature {
    private:
        nda::array<int,2> topology; 

        nda::vector<int> prefactor_Ksigns; 
        nda::vector<int> prefactor_Kexps; 

        nda::array<int,2> vertices;
        nda::array<int,2> edges;
        nda::vector<int> fb; 
        nda::vector<int> pole_inds; 

    public:
        int m; // order
        int n; // number of state variables
        int prefactor_sign; 
        void set_directions(nda::vector_const_view<int> fb);
        void reset_directions(); 
        void set_pole_inds(nda::vector_const_view<int> pole_inds, nda::vector_const_view<double> dlr_rf); 
        void reset_pole_inds(); 
        void set_states(nda::vector_const_view<int> states);
        void reset_states(); 

        int get_prefactor_Ksign(int i); 
        int get_prefactor_Kexp(int i); 

        int get_vertex(int num, int i);
        int get_edge(int num, int pole_ind);
        int get_topology(int i, int j);
        int get_pole_ind(int i); 

    /**
     * @brief Constructor for BackboneSignature
     * 
     * @param[in] topology list of vertices connected by a hybridization line
     * @param[in] n number of state variables
     */
    BackboneSignature(nda::array<int,2> topology, int n);
};

/**
 * @brief Print BackboneSignature to output stream
 * @param[in] os output stream
 * @param[in] B BackboneSignature
 */
std::ostream& operator<<(std::ostream& os, BackboneSignature &B);

/**
 * @brief Multiply by a single vertex in a backbone diagram
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] Fs F operators
 * @param[in] F_dags F^dag operators
 * @param[in] Fdagbars F^{bar dag} operators
 * @param[in] Fbarsrefl F^bar operators
 * @param[in] v_ix vertex index to multiply
 * @param[in] s_ix state index for F operator
 * @param[in] l_ix DLR/AAA pole index
 * @param[in] pole DLR/AAA pole
 * @param[in] T array on which to left-multiply vertex
 */
void multiply_vertex_dense(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    nda::array_const_view<dcomplex,3> Fs, 
    nda::array_const_view<dcomplex,3> F_dags, 
    nda::array_const_view<dcomplex,4> Fdagbars, 
    nda::array_const_view<dcomplex,4> Fbarsrefl, 
    int v_ix, 
    nda::array_view<dcomplex,3> T); 

/**
 * @brief Compute a single edge in a backbone diagram
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Gt Green's function
 * @param[in] e_ix edge index to compute
 * @param[in] GKt array for storing result
 */
void compute_edge_dense(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    nda::array_const_view<dcomplex,3> Gt, 
    int e_ix, 
    nda::array_view<dcomplex,3> GKt); 

/**
 * @brief Evaluate a single backbone diagram
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fs annihilation operators
 * @param[in] F_dags creation operators
 */
nda::array<dcomplex, 3> eval_backbone_dense(BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags);
