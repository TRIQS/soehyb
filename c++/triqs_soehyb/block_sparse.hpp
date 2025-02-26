#pragma once
#include "nda/nda.hpp"
#include "strong_cpl.hpp"
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <vector>

using namespace nda;

/**
  * @class DiagonalOperator
  * @brief Block-sparse storage of block-diagonal operator (e.g. Green's f'n)
  * */
class DiagonalOperator {
    public: 
    std::vector<nda::array<dcomplex,3>> blocks;
    // Green's f'n diagonal blocks
    int num_blocks = blocks.size();

    /**
     * @brief Constructor for DiagonalOperator
     * @param[in] blocks vector of diagonal blocks
     */
    DiagonalOperator(std::vector<nda::array<dcomplex,3>> &blocks);
};

/**
  * @class FOperator
  * @brief Block-sparse storage of F or F^dagger
  * */
class FOperator {
    public: 
    // std::vector<std::tuple<int,nda::array_const_view<dcomplex,2>>> &block_tuples;
    nda::vector<int> block_indices;
    std::vector<nda::array<dcomplex,2>> blocks;
    int num_blocks = block_indices.size();

    /**
     * @brief Constructor for FOperator
     * @param[in] blocks vector of tuples of block-row-indices and corresponding blocks
     */
    FOperator(nda::vector<int> &block_indices, 
        std::vector<nda::array<dcomplex,2>> &blocks);
};

/**
  * @class BlockOperator
  * @brief Block-sparse storage of an arbitrary time-dependent operator
  * */
class BlockOperator {
    public: 
    // block-row block-index and blocks
    // std::vector<std::tuple<int,nda::array_const_view<dcomplex,3>>> &block_tuples;
    nda::vector<int> block_indices;
    std::vector<nda::array<dcomplex,3>> blocks;
    int num_blocks = block_indices.size();

    /**
     * @brief Constructor for BlockOperator
     * @param[in] blocks vector of tuples of block-row-indices and corresponding blocks
     */
    BlockOperator(nda::vector<int> &block_indices, 
        std::vector<nda::array<dcomplex,3>> &blocks);
};

/**
  * @brief Print DiagonalOperator to output stream
  * @param[in] os output stream
  * @param[in] D DiagonalOperator
 */
std::ostream& operator<<(std::ostream& os, DiagonalOperator &D);

/**
  * @brief Print FOperator to output stream
  * @param[in] os output stream
  * @param[in] F FOperator
  */
std::ostream& operator<<(std::ostream& os, FOperator &F);

/**
  * @brief Compute the adjoint of an FOperator
  * @param[in] F FOperator
  * @return F^dagger operator
  * */
FOperator dagger_bs(FOperator &F);

/**
  * @brief Evaluate matrix products in NCA using block-sparse storage
  * @param[in] hyb_self hyb_F object for Delta(t)
  * @param[in] hyb_reflect hyb_F object for Delta(beta-t)
  * @param[in] Gt Greens function
  * @param[in] F F operator
  * @param[in] F_dag F^dagger operator
  * @return NCA self-energy
  * */
// BlockOperator NCA_bs(hyb_F &hyb_self, hyb_F &hyb_reflect, GreensFunction &Gt, 
DiagonalOperator NCA_bs(DiagonalOperator &Gt, 
    std::initializer_list<FOperator> F_list);