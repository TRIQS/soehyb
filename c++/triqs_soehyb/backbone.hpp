#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>
#include <nda/nda.hpp>

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
 * @brief Print Backbone to output stream
 * @param[in] os output stream
 * @param[in] B Backbone
 */
std::ostream &operator<<(std::ostream &os, Backbone &B);