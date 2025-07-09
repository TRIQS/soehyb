#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <nda/nda.hpp>
#include <triqs_soehyb/backbone.hpp>

using namespace nda;

/**
 * @brief Multiply by a block of a single vertex in a backbone diagram
 * @param[in] backbone Backbone object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Fset BlockOpSymQuartet
 * @param[in] v_ix vertex index to multiply
 * @param[in] b_ix block index
 * @param[in] T array on which to left-multiply vertex
 * @param[in] block_dims {# cols vertex, # rows vertex}
 */
void multiply_vertex_block(Backbone &backbone, nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf, BlockOpSymQuartet &Fset,
                           int v_ix, int b_ix, nda::array_view<dcomplex, 3> T, nda::vector_const_view<int> block_dims);

/**
 * @brief Compute a block of a single edge in a backbone diagram
 * @param[in] backbone Backbone object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Gt Green's function
 * @param[in] e_ix edge index to compute
 * @param[in] b_ix block index
 * @param[in] GKt array for storing result
 */
void compute_edge_block(Backbone &backbone, nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf, BlockDiagOpFun &Gt,
                        int e_ix, int b_ix, nda::array_view<dcomplex, 3> GKt_b);

/**
 * @brief Convolve with a block of a single edge in a backbone diagram
 * @param[in] backbone Backbone object
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
void compose_with_edge_block(Backbone &backbone, imtime_ops &itops, nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf,
                             double beta, BlockDiagOpFun &Gt, int e_ix, int b_ix, nda::array_view<dcomplex, 3> T, nda::array_view<dcomplex, 3> GKt_b);

/**
 * @brief Evaluate a single backbone diagram in block-sparse storage
 * @param[in] backbone Backbone object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fset BlockOpSymQuartet
 */
BlockDiagOpFun eval_backbone(Backbone &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb, BlockDiagOpFun &Gt,
                             BlockOpSymQuartet &Fset);