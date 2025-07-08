#include <nda/declarations.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <nda/nda.hpp>

using namespace nda;

// TODO: move documentation from cpp to hpp

/**
 * @class BackboneVertex
 * @brief Abstract representation of a backbone vertex
 */
 // TODO: describe in a bit more detail conceptually what this class is
 // TODO: change name of "pole prime" index to "hybridization index" throughout
class BackboneVertex {
  private:
  bool bar;       // true if the F on this vertex has a bar
  bool dag;       // true if the F on this vertex has a dagger
  int pole_prime; // which of l, l`, ... is associated with the K (and possibly F^bar) on this vertex, i.e., the number of primes on l
  int Ksign;      // 1 if K^+, -1 if K^-, 0 if no K
  int orb;        // value of orbital index on this vertex

  public:
  bool has_bar();
  bool has_dag();
  int get_pole_prime();
  int get_Ksign();
  int get_orb();

  void set_bar(bool b);
  void set_dag(bool b);
  void set_pole_prime(int i);
  void set_Ksign(int i);
  void set_orb(int i);

  /**
     * @brief Constructor for BackboneVertex
     */
  BackboneVertex();
};

/**
 * @class BackboneSignature
 * @brief Abstract representation of a backbone diagram
 */
 // TODO: describe in a bit more detail conceptually what this class is
 // TODO: rename BackboneSignature class to Backbone
class BackboneSignature {
  private:
  nda::array<int, 2> topology;

  nda::vector<int> prefactor_Ksigns;
  nda::vector<int> prefactor_Kexps;
  nda::array<int, 2> edges;
  nda::vector<int> fb;
  nda::vector<int> pole_inds;

  public:
  int m; // order
  int n; // number of orbital indices
  int prefactor_sign;
  std::vector<BackboneVertex> vertices;

  void set_directions(nda::vector_const_view<int> fb);
  void reset_directions();
  void set_pole_inds(nda::vector_const_view<int> pole_inds, nda::vector_const_view<double> dlr_rf);
  void reset_pole_inds();
  void set_orb_inds(nda::vector_const_view<int> orb_inds);
  void reset_orb_inds();

  int get_prefactor_Ksign(int i);
  int get_prefactor_Kexp(int i);

  bool has_vertex_bar(int i);
  bool has_vertex_dag(int i);
  int get_vertex_pole_prime(int i);
  int get_vertex_Ksign(int i);
  int get_vertex_orb(int i);

  int get_edge(int num, int pole_ind);
  int get_topology(int i, int j);
  int get_pole_ind(int i);

  /**
     * @brief Constructor for BackboneSignature
     * 
     * @param[in] topology list of vertices connected by a hybridization line
     * @param[in] n number of orbital indices
     */
  BackboneSignature(nda::array<int, 2> topology, int n);
};

/**
 * @brief Print BackboneSignature to output stream
 * @param[in] os output stream
 * @param[in] B BackboneSignature
 */
std::ostream &operator<<(std::ostream &os, BackboneSignature &B);

/**
 * @brief Multiply by a single vertex in a backbone diagram using dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Fset DenseFSet
 * @param[in] v_ix vertex index to multiply
 * @param[in] T array on which to left-multiply vertex
 */
void multiply_vertex_dense(BackboneSignature &backbone, nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf, DenseFSet &Fset,
                           int v_ix, nda::array_view<dcomplex, 3> T);

// TODO: merge compute_edge_dense into compose_with_edge_dense

/**
 * @brief Compute a single edge in a backbone diagram using dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Gt Green's function
 * @param[in] e_ix edge index to compute
 * @param[in] GKt array for storing result
 */
void compute_edge_dense(BackboneSignature &backbone, nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf,
                        nda::array_const_view<dcomplex, 3> Gt, int e_ix, nda::array_view<dcomplex, 3> GKt);

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
void compose_with_edge_dense(BackboneSignature &backbone, imtime_ops &itops, nda::vector_const_view<double> dlr_it,
                             nda::vector_const_view<double> dlr_rf, double beta, nda::array_const_view<dcomplex, 3> Gt, int e_ix,
                             nda::array_view<dcomplex, 3> T, nda::array_view<dcomplex, 3> GKt);

/**
 * @brief Multiply by the zero vertex and the vertex connected to zero
 * @param[in] backbone BackboneSignature object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] is_forward true if the line connected to the zero vertex is forward in time
 * @param[in] Fset DenseFSet
 * @param[in] Tkaps intermediate storage array
 * @param[in] Tmu intermediate storage array
 * @param[in] T array for storing result
 */
void multiply_zero_vertex(BackboneSignature &backbone, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                          bool is_forward, DenseFSet &Fset, nda::array_view<dcomplex, 4> Tkaps, nda::array_view<dcomplex, 3> Tmu,
                          nda::array_view<dcomplex, 3> T);

/**
 * @brief Evaluate a backbone diagram for particular orbital indices, poles, and line directions in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] Gt Greens function
 * @param[in] Fset DenseFSet (cre/ann operators with and without bars)
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 * @param[in] Tkaps intermediate storage array
 * @param[in] Tmu intermediate storage array
 */
void eval_backbone_fixed_orbs_poles_lines_dense(BackboneSignature &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                                nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset, nda::array_view<dcomplex, 3> hyb_refl,
                                                nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf,
                                                nda::array_view<dcomplex, 3> T, nda::array_view<dcomplex, 3> GKt, nda::array_view<dcomplex, 4> Tkaps,
                                                nda::array_view<dcomplex, 3> Tmu);

/**
 * @brief Evaluate a backbone diagram with fixed poles, and line directions in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] Gt Greens function
 * @param[in] Fset DenseFSet (cre/ann operators with and without bars)
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 * @param[in] Tkaps intermediate storage array
 * @param[in] Tmu intermediate storage array
 * @param[in] orb_inds vector of orbital indices
 * @param[in] Sigma_L array for storing backbone result, including prefactor, over all orbital indices
 */
void eval_backbone_fixed_poles_lines_dense(BackboneSignature &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                           nda::array_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset,
                                           nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf,
                                           nda::array_view<dcomplex, 3> T, nda::array_view<dcomplex, 3> GKt, nda::array_view<dcomplex, 4> Tkaps,
                                           nda::array_view<dcomplex, 3> Tmu, nda::vector_view<int> orb_inds, nda::array_view<dcomplex, 3> Sigma_L);

/**
 * @brief Evaluate a backbone diagram with fixed line directions in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] Gt Greens function
 * @param[in] Fset DenseFSet (cre/ann operators with and without bars)
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 * @param[in] Tkaps intermediate storage array
 * @param[in] Tmu intermediate storage array
 * @param[in] orb_inds vector of orbital indices
 * @param[in] Sigma_L array for storing backbone result, including prefactor, over all orbital indices
 * @param[in] pole_inds vector of pole indices (pole_inds)
 * @param[in] sign +/-1, depending on diagram order and line directions
 * @param[in] Sigma array for storing self-energy contribution 
 */
void eval_backbone_fixed_lines_dense(BackboneSignature &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                     nda::array_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset,
                                     nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf, nda::array_view<dcomplex, 3> T,
                                     nda::array_view<dcomplex, 3> GKt, nda::array_view<dcomplex, 4> Tkaps, nda::array_view<dcomplex, 3> Tmu,
                                     nda::vector_view<int> orb_inds, nda::array_view<dcomplex, 3> Sigma_L, nda::vector_view<int> pole_inds, int sign,
                                     nda::array_view<dcomplex, 3> Sigma);

/**
 * @brief Evaluate a single backbone diagram in dense storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fset DenseFSet
 */
nda::array<dcomplex, 3> eval_backbone_dense(BackboneSignature &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                            nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset);

/**
 * @brief Multiply by a block of a single vertex in a backbone diagram
 * @param[in] backbone BackboneSignature object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Fset BlockOpSymQuartet
 * @param[in] v_ix vertex index to multiply
 * @param[in] b_ix block index
 * @param[in] T array on which to left-multiply vertex
 * @param[in] block_dims {# cols vertex, # rows vertex}
 */
void multiply_vertex_block(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    BlockOpSymQuartet& Fset, 
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
void compute_edge_block(BackboneSignature &backbone, nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf, BlockDiagOpFun &Gt,
                        int e_ix, int b_ix, nda::array_view<dcomplex, 3> GKt_b);

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
void compose_with_edge_block(BackboneSignature &backbone, imtime_ops &itops, nda::vector_const_view<double> dlr_it,
                             nda::vector_const_view<double> dlr_rf, double beta, BlockDiagOpFun &Gt, int e_ix, int b_ix,
                             nda::array_view<dcomplex, 3> T, nda::array_view<dcomplex, 3> GKt_b);

/**
 * @brief Evaluate a single backbone diagram in block-sparse storage
 * @param[in] backbone BackboneSignature object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fset BlockOpSymQuartet
 */
BlockDiagOpFun eval_backbone(
    BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    BlockDiagOpFun& Gt, 
    BlockOpSymQuartet& Fset);