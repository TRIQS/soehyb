#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <nda/nda.hpp>

using namespace nda;

/**
 * @class BackboneVertex
 * @brief Abstract representation of a backbone vertex
 * @note This is a lightweight class used to represent a vertex in a backbone diagram. Specifically, it says whether this vertex has a creation 
 * @note operator, an annihilation operator, or a linear combination of one of these with coefficients from a decomposition of a hybridization
 * @note function. It also stores which hybridization index (l, l`, ...) is associated with the K on this vertex, the sign on K, and the orbital
 * @note index on the operator. All of this information is specified by integer indices, since evaluation is left to the DiagramEvaluator class. 
 */
class BackboneVertex {
  private:
  bool bar;    // true if the F on this vertex has a bar
  bool dag;    // true if the F on this vertex has a dagger
  int hyb_ind; // which of l, l`, ... is associated with the K (and possibly F^bar) on this vertex, i.e., the number of primes on l
  int Ksign;   // 1 if K^+, -1 if K^-, 0 if no K
  int orb;     // value of orbital index on this vertex

  public:
  bool has_bar();
  bool has_dag();
  int get_hyb_ind();
  int get_Ksign();
  int get_orb();

  void set_bar(bool b);
  void set_dag(bool b);
  void set_hyb_ind(int i);
  void set_Ksign(int i);
  void set_orb(int i);

  /**
   * @brief Constructor for BackboneVertex
   */
  BackboneVertex();
};

/**
 * @class Backbone
 * @brief Abstract representation of a backbone diagram
 * @note This is a lightweight class used to represent a backbone diagram of a specific order and topology. Its attributes contain information about 
 * @note the prefactor; the locations, signs, and hybridization indices of the K's on the edges; and the vertices (see documentation for 
 * @note BackboneVertex). For a given order and topology, the aforementioned information is fixed by the directions of the hybridization lines,
 * @note the signs of the values of the hybridization indices, and the orbital indices on the vertices. These can be set and reset by methods of this 
 * @note class so that a single Backbone object can be reused for all backbones of a given order and topology. All of this information is 
 * @note specified by integer and Boolean indices, since evaluation is left to the DiagramEvaluator class.
 */
class Backbone {
  private:
  nda::array<int, 2> topology;

  nda::vector<int> prefactor_Ksigns; // for each of m-1 pole indices (l, l`, ...), the sign on K_l^?(0)
  nda::vector<int> prefactor_Kexps;  // for each of m-1 pole_indices, the exponent on K_l(0)^?
  nda::array<int, 2> edges;
  // all entries are +/-1 or 0
  // edges(e,p) = if +/-1, +/- on K_l^?(tau) on edge e, where l has p primes
  // else, edges(e,p) = 0, and there is not K_l^?(tau) on edge e
  // Example: if edges(3,2) = -1, then the third edge has K_{l''}^{-1}(tau)
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
  nda::vector<int> fb;        // directions of the hybridization lines, 0 for backward, 1 for forward
  nda::vector<int> pole_inds; // values of the hybridization indices (i.e. values of l, l`, ...)

  public:
  int m;              // order
  int n;              // number of orbital indices
  int prefactor_sign; // +1 or -1, depending on the sign of the prefactor
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
  int get_vertex_hyb_ind(int i);
  int get_vertex_Ksign(int i);
  int get_vertex_orb(int i);

  int get_edge(int num, int pole_ind);
  int get_topology(int i, int j);
  int get_pole_ind(int i);
  int get_fb(int i); 

  /**
   * @brief Constructor for Backbone
   * 
   * @param[in] topology list of vertices connected by a hybridization line
   * @param[in] n number of orbital indices
   */
  Backbone(nda::array<int, 2> topology, int n);
};

/**
 * @class DiagramEvaluator
 * @brief Class for evaluating a diagram of a given order and topology
 * @note This class is used to evaluate all the backbone decompositions of a given order and topology. It reads the information from a Backbone object
 * @note and contains the Green's functions and creation/annihilation operators needed to actually compute the diagram. It also contains temporary 
 * @note arrays required for computation. 
 */
class DiagramEvaluator {
  public:
  double beta;                      // inverse temperature
  imtime_ops itops;                 // DLR imaginary time object
  nda::array<dcomplex, 3> hyb;      // hybridization function at imaginary time nodes
  nda::array<dcomplex, 3> hyb_refl; // hybridization function at (beta - tau) nodes
  nda::array<dcomplex, 3> Gt;       // Green's function at imaginary time nodes
  DenseFSet Fset;                   // DenseFSet (cre/ann operators with and without bars)
  nda::vector<double> dlr_it;       // DLR imaginary time nodes in relative ordering
  nda::vector<double> dlr_rf;       // DLR frequency nodes
  nda::array<dcomplex, 3> Sigma;    // array for storing self-energy contribution (final result)
  nda::array<dcomplex, 3> T;        // array for storing intermediate result
  nda::array<dcomplex, 3> GKt;      // array for storing result of edge computation
  nda::array<dcomplex, 4> Tkaps;    // intermediate storage array
  nda::array<dcomplex, 3> Tmu;      // intermediate storage array
  nda::array<dcomplex, 3> Sigma_L;  // intermediate storage array for backbone result, including prefactor, over all orbital indices

  void reset(); // reset all arrays to zero

  /**
   * @brief Constructor for DiagramEvaluator
   * 
   * @param[in] beta inverse temperature
   * @param[in] itops DLR imaginary time object
   * @param[in] hyb hybridization function at imaginary time nodes
   * @param[in] hyb_refl hybridization function at (beta - tau) nodes
   * @param[in] Gt Green's function at imaginary time nodes
   * @param[in] Fset DenseFSet (cre/ann operators with and without bars)
   */
  DiagramEvaluator(double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                   nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset);
};

/**
 * @brief Multiply by a single vertex in a backbone diagram using dense storage
 * @param[in] backbone Backbone object
 * @param[in] diag_eval DiagramEvaluator object
  * @param[in] v_ix vertex index to multiply
 */
void multiply_vertex_dense(Backbone &backbone, DiagramEvaluator &diag_eval, int v_ix);

/**
 * @brief Convolve with a single edge in a backbone diagram using dense storage
 * @param[in] backbone Backbone object
 * @param[in] diag_eval DiagramEvaluator object
 * @param[in] e_ix edge index to compute
 */
void compose_with_edge_dense(Backbone &backbone, DiagramEvaluator &diag_eval, int e_ix);

/**
 * @brief Multiply by the zero vertex and the vertex connected to zero
 * @param[in] backbone Backbone object
 * @param[in] diag_eval DiagramEvaluator object
 * @param[in] is_forward true if the line connected to the zero vertex is forward in time
 */
void multiply_zero_vertex(Backbone &backbone, DiagramEvaluator &diag_eval, bool is_forward);

/**
 * @brief Evaluate a diagram of a given order and topolgoy in dense storage
 * @param[in] backbone Backbone object
 * @param[in] diag_eval DiagramEvaluator object
 */
nda::array<dcomplex, 3> eval_diagram_dense(Backbone &backbone, DiagramEvaluator &diag_eval);

/**
 * @brief Evaluate a backbone diagram with fixed line directions in dense storage
 * @param[in] backbone Backbone object
 * @param[in] diag_eval DiagramEvaluator object
 */
void eval_diagram_fixed_lines_dense(Backbone &backbone, DiagramEvaluator &diag_eval);

/**
 * @brief Evaluate a backbone diagram with fixed poles and line directions in dense storage
 * @param[in] backbone Backbone object
 * @param[in] diag_eval DiagramEvaluator object
 */
void eval_diagram_fixed_poles_lines_dense(Backbone &backbone, DiagramEvaluator &diag_eval);

/**
 * @brief Evaluate a backbone diagram with fixed orbital indices, poles, and line directions in dense storage
 * @param[in] backbone Backbone object
 * @param[in] diag_eval DiagramEvaluator object
 */
void eval_diagram_fixed_orbs_poles_lines_dense(Backbone &backbone, DiagramEvaluator &diag_eval);

/**
 * @brief Multiply by a single vertex in a backbone diagram using dense storage
 * @param[in] backbone Backbone object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] Fset DenseFSet
 * @param[in] v_ix vertex index to multiply
 * @param[in] T array on which to left-multiply vertex
 */
void multiply_vertex_dense(Backbone &backbone, nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf, DenseFSet &Fset,
                           int v_ix, nda::array_view<dcomplex, 3> T);

/**
 * @brief Convolve with a single edge in a backbone diagram using dense storage
 * @param[in] backbone Backbone object
 * @param[in] itops DLR imaginary time object
 * @param[in] dlr_it DLR imaginary time nodes in relative ordering
 * @param[in] dlr_rf DLR frequency nodes
 * @param[in] beta inverse temperature
 * @param[in] Gt Green's function
 * @param[in] e_ix edge index to compute
 * @param[in] T array for storing result
 * @param[in] GKt array for storing result of edge computation
 */
void compose_with_edge_dense(Backbone &backbone, imtime_ops &itops, nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf,
                             double beta, nda::array_const_view<dcomplex, 3> Gt, int e_ix, nda::array_view<dcomplex, 3> T,
                             nda::array_view<dcomplex, 3> GKt);

/**
 * @brief Multiply by the zero vertex and the vertex connected to zero
 * @param[in] backbone Backbone object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] is_forward true if the line connected to the zero vertex is forward in time
 * @param[in] Fset DenseFSet
 * @param[in] Tkaps intermediate storage array
 * @param[in] Tmu intermediate storage array
 * @param[in] T array for storing result
 */
void multiply_zero_vertex(Backbone &backbone, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl, bool is_forward,
                          DenseFSet &Fset, nda::array_view<dcomplex, 4> Tkaps, nda::array_view<dcomplex, 3> Tmu, nda::array_view<dcomplex, 3> T);

/**
 * @brief Evaluate a single backbone diagram in dense storage
 * @param[in] backbone Backbone object
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] Gt Greens function
 * @param[in] Fset DenseFSet
 */
nda::array<dcomplex, 3> eval_backbone_dense(Backbone &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                            nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset);

/**
 * @brief Evaluate a backbone diagram with fixed line directions in dense storage
 * @param[in] backbone Backbone object
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
void eval_backbone_fixed_lines_dense(Backbone &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                     nda::array_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset,
                                     nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf, nda::array_view<dcomplex, 3> T,
                                     nda::array_view<dcomplex, 3> GKt, nda::array_view<dcomplex, 4> Tkaps, nda::array_view<dcomplex, 3> Tmu,
                                     nda::vector_view<int> orb_inds, nda::array_view<dcomplex, 3> Sigma_L, nda::vector_view<int> pole_inds, int sign,
                                     nda::array_view<dcomplex, 3> Sigma);

/**
 * @brief Evaluate a backbone diagram with fixed poles and line directions in dense storage
 * @param[in] backbone Backbone object
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
void eval_backbone_fixed_poles_lines_dense(Backbone &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                           nda::array_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset,
                                           nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf,
                                           nda::array_view<dcomplex, 3> T, nda::array_view<dcomplex, 3> GKt, nda::array_view<dcomplex, 4> Tkaps,
                                           nda::array_view<dcomplex, 3> Tmu, nda::vector_view<int> orb_inds, nda::array_view<dcomplex, 3> Sigma_L);

/**
 * @brief Evaluate a backbone diagram for particular orbital indices, poles, and line directions in dense storage
 * @param[in] backbone Backbone object
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
void eval_backbone_fixed_orbs_poles_lines_dense(Backbone &backbone, double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                                nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset, nda::array_view<dcomplex, 3> hyb_refl,
                                                nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf,
                                                nda::array_view<dcomplex, 3> T, nda::array_view<dcomplex, 3> GKt, nda::array_view<dcomplex, 4> Tkaps,
                                                nda::array_view<dcomplex, 3> Tmu);

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

/**
 * @brief Print Backbone to output stream
 * @param[in] os output stream
 * @param[in] B Backbone
 */
std::ostream &operator<<(std::ostream &os, Backbone &B);