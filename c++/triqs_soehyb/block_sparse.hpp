#pragma once
#include "nda/nda.hpp"
#include "strong_cpl.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <vector>

using namespace nda;

// TODO templates for double/dcomplex

/**
 * @class BlockDiagOpFun (BDOF)
 * @brief Block-sparse storage of time-dependent block-diagonal operator (e.g. Green's f'n)
 */
class BlockDiagOpFun {
  private:
  std::vector<nda::array<dcomplex, 3>> blocks;
  int num_block_cols;
  nda::vector<int> zero_block_indices;

  public:
  BlockDiagOpFun &operator+=(const BlockDiagOpFun &G);
  void set_blocks(std::vector<nda::array<dcomplex, 3>> &blocks);
  void set_block(int i, nda::array_const_view<dcomplex, 3> block);
  const std::vector<nda::array<dcomplex, 3>> &get_blocks() const;
  nda::array_const_view<dcomplex, 3> get_block(int i) const;
  nda::vector<int> get_block_sizes() const;
  int get_block_size(int i) const;
  int get_max_block_size() const;
  int get_num_block_cols() const;
  int get_zero_block_index(int i) const;
  int get_num_time_nodes() const;
  void add_block(int i, nda::array_const_view<dcomplex, 3> block);
  static std::string hdf5_format();
  friend void h5_write(h5::group g, const std::string &subgroup_name, const BlockDiagOpFun &BDOF);
  friend void h5_read(h5::group g, const std::string &subgroup_name, BlockDiagOpFun &BDOF);

  /**
     * @brief Constructor for BlockDiagOpFun
     * @param[in] blocks vector of diagonal blocks
     * @param[in] zero_block_indices if i-th entry is -1, then blocks(i) = 0
     */
  BlockDiagOpFun(std::vector<nda::array<dcomplex, 3>> &blocks, nda::vector_const_view<int> zero_block_indices);

  /**
     * @brief Constructor for BlockDiagOpFun with blocks of zeros
     * @param[in] r number of imaginary time nodes
     * @param[in] block_sizes vector of sizes of diagonal blocks
     */
  BlockDiagOpFun(int r, nda::vector_const_view<int> block_sizes);
};

/**
 * @class BlockOp (BO)
 * @brief Block-sparse storage of F or F^dagger
 */
class BlockOp {
  private:
  nda::vector<int> block_indices;
  std::vector<nda::array<dcomplex, 2>> blocks;
  int num_block_cols;

  public:
  BlockOp &operator+=(const BlockOp &F);
  void set_block_indices(nda::vector<int> &block_indices);
  void set_block_index(int i, int block_index);
  void set_blocks(std::vector<nda::array<dcomplex, 2>> &blocks);
  void set_block(int i, nda::array_const_view<dcomplex, 2> block);
  nda::vector_const_view<int> get_block_indices() const;
  int get_block_index(int i) const;
  const std::vector<nda::array<dcomplex, 2>> &get_blocks() const;
  nda::array_const_view<dcomplex, 2> get_block(int i) const;
  int get_num_block_cols() const;
  nda::array<int, 2> get_block_sizes() const;
  nda::vector<int> get_block_size(int i) const;
  int get_block_size(int block_ind, int dim) const;

  /**
     * @brief Constructor for BlockOp
     * @param[in] block_indices vector of block-column indices in each block-col
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in col i
     */
  BlockOp(nda::vector<int> &block_indices, std::vector<nda::array<dcomplex, 2>> &blocks);

  /**
     * @brief Constructor for BlockOp with blocks of zeros
     * @param[in] block_indices vector of block-column indices in each block-col
     * @param[in] block_sizes array of block sizes
     * @note block_indices[i] = -1 if F does not have a block in col i
     */
  BlockOp(nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
};

/**
 * @class BlockOp3D 
 * @brief Abstract superclass for block-sparse storage of sequences of matrices with the same sparsity pattern
 */
class BlockOp3D {
  protected:
  nda::vector<int> block_indices;
  std::vector<nda::array<dcomplex, 3>> blocks;
  int num_block_cols;

  public:
  void set_block_indices(nda::vector<int> &block_indices);
  void set_block_index(int i, int block_index);
  void set_blocks(std::vector<nda::array<dcomplex, 3>> &blocks);
  void set_block(int i, nda::array_const_view<dcomplex, 3> block);
  nda::vector_const_view<int> get_block_indices() const;
  int get_block_index(int i) const;
  const std::vector<nda::array<dcomplex, 3>> &get_blocks() const;
  nda::array_const_view<dcomplex, 3> get_block(int i) const;
  int get_num_block_cols() const;
  nda::array<int, 2> get_block_sizes() const;
  nda::vector<int> get_block_size(int i) const;

  /**
   * @brief Constructor for BlockOpFun 
   * @param[in] block_indices vector of block-col-indices
   * @param[in] blocks vector of blocks
   * @note block_indices[i] = -1 if F does not have a block in col i
   */
  BlockOp3D(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks);

  /**
   * @brief Constructor for BlockOpFun with blocks of zeros
   * @param[in] r number of imaginary time nodes
   * @param[in] block_indices vector of block-col-indices
   * @param[in] block_sizes vector of sizes of blocks
   */
  BlockOp3D(int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
};

/**
 * @class BlockOpFun (BOF)
 * @brief Block-sparse storage of an arbitrary time-dependent operator
 */
class BlockOpFun : public BlockOp3D {
  public:
  BlockOpFun &operator+=(const BlockOpFun &A);
  int get_num_time_nodes() const;

  /**
     * @brief Constructor for BlockOpFun 
     * @param[in] block_indices vector of block-col-indices
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in col i
     */
  BlockOpFun(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks);

  /**
     * @brief Constructor for BlockOpFun with blocks of zeros
     * @param[in] r number of imaginary time nodes
     * @param[in] block_indices vector of block-col-indices
     * @param[in] block_sizes vector of sizes of blocks
     */
  BlockOpFun(int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
};

/**
 * @class DenseFSet
 * @brief Container for (linear combinations of) creation and annihilation operators in dense storage
 */
class DenseFSet {
  public:
  nda::array<dcomplex, 3> Fs;
  nda::array<dcomplex, 3> F_dags;
  nda::array<dcomplex, 4> F_dag_bars;
  nda::array<dcomplex, 4> F_bars_refl;

  /**
   * @brief Constructor for DenseFSet
   * @param[in] Fs annihilation operators
   * @param[in] F_dags creation operators
   * @param[in] hyb_coeffs DLR coefficients of hybridization
   * @param[in] hyb_refl_coeffs DLR coefficients of reflected hybridization
   */
  DenseFSet(nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags, nda::array_const_view<dcomplex, 3> hyb_coeffs,
            nda::array_const_view<dcomplex, 3> hyb_refl_coeffs);
};

/**
 * @class BlockOpSymSet (BOSS)
 * @brief Container for (linear combinations of) creation/annihilation operators with the same block-sparse structure
 */
class BlockOpSymSet : public BlockOp3D {
  public:
  int get_size_sym_set() const;

  /**
   * @brief Constructor for BlockOpSymSet
   * @param[in] block_indices vector of block-col-indices
   * @param[in] blocks vector of blocks
   * @note block_indices[i] = -1 if F does not have a block in col i
   */
  BlockOpSymSet(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks);

  /**
   * @brief Constructor for BlockOpSymSet with blocks of zeros
   * @param[in] q number of operators in this set
   * @param[in] block_indices vector of block-col-indices
   * @param[in] block_sizes vector of sizes of blocks
   */
  BlockOpSymSet(int q, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
};



/**
 * @class BlockOpSymSetBar
 * @brief Class for block-sparse storage of a set of F^bar operators with the same sparsity pattern
 */
class BlockOpSymSetBar {
  protected:
  nda::vector<int> block_indices;
  std::vector<nda::array<dcomplex, 4>> blocks;
  int num_block_cols;

  public:
  void set_block_indices(nda::vector<int> &block_indices);
  void set_block_index(int i, int block_index);
  void set_blocks(std::vector<nda::array<dcomplex, 4>> &blocks);
  void set_block(int i, nda::array_const_view<dcomplex, 4> block);
  nda::vector_const_view<int> get_block_indices() const;
  int get_block_index(int i) const;
  const std::vector<nda::array<dcomplex, 4>> &get_blocks() const;
  nda::array_const_view<dcomplex, 4> get_block(int i) const;
  int get_num_block_cols() const;
  int get_size_sym_set() const;
  int get_num_time_nodes() const;
  void add_block(int i, int s, int t, nda::array_const_view<dcomplex, 2> block); // add to block i, symmetry index s, time index t
  
  /**
   * @brief Constructor for BlockOpSymSetBar
   * @param[in] block_indices vector of block-col-indices
   * @param[in] blocks vector of blocks
   * @note block_indices[i] = -1 if F does not have a block in col i
   */
  BlockOpSymSetBar(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 4>> &blocks);

  /**
   * @brief Constructor for BlockOpSymSetBar with blocks of zeros
   * @param[in] q number of orbital indices associated with the block-sparse structure
   * @param[in] r rank of the DLR imaginary time object
   * @param[in] block_indices vector of block-col-indices
   * @param[in] block_sizes vector of sizes of blocks
   */
  BlockOpSymSetBar(int q, int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
};

/**
 * @class BlockOpSymQuartet (BOSQ)
 * @brief Container for multiple symmetry sets of BOSS 
 */
class BlockOpSymQuartet {
  public:
  std::vector<BlockOpSymSet> Fs;
  std::vector<BlockOpSymSet> F_dags;
  std::vector<BlockOpSymSetBar> F_dag_bars;
  std::vector<BlockOpSymSetBar> F_bars_refl;

  /** 
   * @brief Constructor for BlockOpSymQuartet
   * @param[in] Fs vector of annihilation operator BOSS
   * @param[in] F_dags vector of creation operator BOSS
   * @param[in] F_dag_bars vector of vectors of linear combinations of creation operator BOSS 
   * @param[in] F_bars_refl vector of vectors of linear combinations of annihilation operator BOSS
   */
   BlockOpSymQuartet(std::vector<BlockOpSymSet> Fs, std::vector<BlockOpSymSet> F_dags, nda::array_const_view<dcomplex, 3> hyb_coeffs, 
                     nda::array_const_view<dcomplex, 3> hyb_refl_coeffs); 
};

/**
 * @brief Print BlockDiagOpFun to output stream
 * @param[in] os output stream
 * @param[in] D BlockDiagOpFun
 */
std::ostream &operator<<(std::ostream &os, BlockDiagOpFun &D);

/**
 * @brief Print BlockOp to output stream
 * @param[in] os output stream
 * @param[in] F BlockOp
 */
std::ostream &operator<<(std::ostream &os, BlockOp &F);

/**
 * @brief Compute the adjoint of a BlockOp
 * @param[in] F BlockOp
 * @return F^dagger operator
 */
BlockOp dagger_bs(BlockOp const &F);

/**
 * @brief Compute a product between a scalar and a BlockOp
 * @param[in] c dcomplex
 * @param[in] F BlockOp
 */
BlockOp operator*(const dcomplex c, const BlockOp &F);

/**
 * @brief Compute a product between a scalar and a BlockOp3D
 * @param[in] c dcomplex
 * @param[in] F BlockOp3D
 */
BlockOp3D operator*(const dcomplex c, const BlockOp3D &F);

/**
 * @brief Convert a BlockOpFun with diagonal structure to a BlockDiagOpFun
 * @param[in] A BlockOpFun
 * @return BlockDiagOpFun
 */
BlockDiagOpFun BOFtoBDOF(BlockOpFun const &A);

/**
 * @brief Compute noninteracting Green's function from Hamiltonian as a BDOF
 * @param[in] H_blocks vector of blocks of Hamiltonian
 * @param[in] H_block_inds vector, -1 if Hamiltonian has zero block in corresponding block column
 * @param[in] beta inverse temperature
 * @param[in] dlr_it imaginary time nodes in absolute format
 */
BlockDiagOpFun nonint_gf_BDOF(std::vector<nda::array<double, 2>> H_blocks, nda::vector<int> H_block_inds, double beta,
                              nda::vector_const_view<double> dlr_it);

/**
 * @brief Compute noninteracting Green's function from Hamiltonian as a BDOF
 * @param[in] H_blocks vector of blocks of Hamiltonian
 * @param[in] H_block_inds vector, -1 if Hamiltonian has zero block in corresponding block column
 * @param[in] beta inverse temperature
 * @param[in] dlr_it imaginary time nodes in absolute format
 */
BlockDiagOpFun nonint_gf_BDOF(std::vector<nda::array<double, 2>> H_blocks, nda::vector<int> H_block_inds, double beta,
                              nda::vector_const_view<double> dlr_it);
/**
 * @brief Compute noninteracting Green's function from Hamiltonian as a BDOF
 * @param[in] H_blocks vector of blocks of Hamiltonian
 * @param[in] H_block_inds vector, -1 if Hamiltonian has zero block in corresponding block column
 * @param[in] beta inverse temperature
 * @param[in] dlr_it imaginary time nodes in absolute format
 */
BlockDiagOpFun nonint_gf_BDOF(std::vector<nda::array<double, 2>> H_blocks, nda::vector<int> H_block_inds, double beta,
                              nda::vector_const_view<double> dlr_it);