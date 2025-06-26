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

        // nda::array<int,2> vertices;
        nda::vector<bool> vertex_bars;
        nda::vector<bool> vertex_dags;
        nda::vector<int> vertex_which_pole_ind;
        nda::vector<int> vertex_Ksigns;
        nda::vector<int> vertex_states; 

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

        bool get_vertex_bar(int i); 
        bool get_vertex_dag(int i); 
        int get_vertex_which_pole_ind(int i); 
        int get_vertex_Ksign(int i); 
        int get_vertex_state(int i); 

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
 * @brief Multiply by a single vertex in a backbone diagram using dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Fs F operators
 * @param[in] F_dags F^dag operators
 * @param[in] Fdagbars F^{bar dag} operators
 * @param[in] Fbarsrefl F^bar operators
 * @param[in] v_ix vertex index to multiply
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
 * @brief Multiply by a single vertex in a backbone diagram using dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Fset DenseFSet
 * @param[in] v_ix vertex index to multiply
 * @param[in] T array on which to left-multiply vertex
 */
void multiply_vertex_dense(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    DenseFSet& Fset, 
    int v_ix, 
    nda::array_view<dcomplex,3> T); 

/**
 * @brief Compute a single edge in a backbone diagram using dense storage
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
 * @brief Convolve with a single edge in a backbone diagram using dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] itops DLR imaginary time object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] beta inverse temperature
 * @param[in] Gt Green's function
 * @param[in] e_ix edge index to compute
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 */
void compose_with_edge_dense(
    BackboneSignature& backbone, 
    imtime_ops& itops, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    double beta, 
    nda::array_const_view<dcomplex,3> Gt, 
    int e_ix, 
    nda::array_view<dcomplex,3> T, 
    nda::array_view<dcomplex,3> GKt
);

/**
 * @brief Evaluate a backbone diagram with fixed states, poles, and line directions in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fs annihilation operators
 * @param[in] F_dags creation operators
 * @param[in] Fdagbars linear combinations of F_dags
 * @param[in] Fbarsrefl linear combinations of Fs
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 * @param[in] Tkaps intermediate storage array
 * @param[in] Tmu intermediate storage array
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 */
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
);

/**
 * @brief Evaluate a backbone diagram with fixed poles, and line directions in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fs annihilation operators
 * @param[in] F_dags creation operators
 * @param[in] Fdagbars linear combinations of F_dags
 * @param[in] Fbarsrefl linear combinations of Fs
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 * @param[in] Tkaps intermediate storage array
 * @param[in] Tmu intermediate storage array
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] states vector of state indices
 * @param[in] Sigma_L array for storing backbone result, including prefactor, over all states
 */
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
);

/**
 * @brief Evaluate a backbone diagram with fixed line directions in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fs annihilation operators
 * @param[in] F_dags creation operators
 * @param[in] Fdagbars linear combinations of F_dags
 * @param[in] Fbarsrefl linear combinations of Fs
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 * @param[in] Tkaps intermediate storage array
 * @param[in] Tmu intermediate storage array
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] states vector of state indices
 * @param[in] Sigma_L array for storing backbone result, including prefactor, over all states
 * @param[in] pole_inds vector of pole indices (pole_inds)
 * @param[in] sign (-1)^(s+f+m)
 * @param[in] Sigma array for storing self-energy contribution 
 */
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
);

/**
 * @brief Evaluate a single backbone diagram in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fs annihilation operators
 * @param[in] F_dags creation operators
 */
nda::array<dcomplex, 3> eval_backbone_dense(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags);

/**
 * @brief Evaluate a single backbone diagram in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fset DenseFSet
 */
nda::array<dcomplex, 3> eval_backbone_dense(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    DenseFSet& Fset); 

/**
 * @brief Multiply by a block of a single vertex in a backbone diagram
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Fs F operators
 * @param[in] F_dags F^dag operators
 * @param[in] Fdagbars F^{bar dag} operators
 * @param[in] Fbarsrefl F^bar operators
 * @param[in] v_ix vertex index to multiply
 * @param[in] b_ix block index
 * @param[in] T array on which to left-multiply vertex
 * @param[in] block_dims {# cols vertex, # rows vertex}
 */
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
    nda::vector_const_view<int> block_dims); 

/**
 * @brief Compute a block of a single edge in a backbone diagram
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Gt Green's function
 * @param[in] e_ix edge index to compute
 * @param[in] b_ix block index
 * @param[in] GKt array for storing result
 */
void compute_edge_block(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    BlockDiagOpFun& Gt, 
    int e_ix, 
    int b_ix, 
    nda::array_view<dcomplex,3> GKt_b); 

/**
 * @brief Convolve with a block of a single edge in a backbone diagram
 * @param[in] backbone BackboneSignature object
 * @param[in] itops DLR imaginary time object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] beta inverse temperature
 * @param[in] Gt Green's function
 * @param[in] e_ix edge index to compute
 * @param[in] b_ix block index
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 */
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
    nda::array_view<dcomplex,3> GKt_b);

/**
 * @brief Evaluate a single backbone diagram in block-sparse storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fs annihilation operators
 * @param[in] F_dags creation operators
 */
BlockDiagOpFun eval_backbone(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    BlockDiagOpFun& Gt, 
    std::vector<BlockOp>& Fs, 
    std::vector<BlockOp>& F_dags);