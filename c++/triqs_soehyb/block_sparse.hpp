#pragma once
#include "nda/nda.hpp"
#include "strong_cpl.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <vector>

using namespace nda;

/**
 * @class DiagonalOperator
 * @brief Block-sparse storage of block-diagonal operator (e.g. Green's f'n)
 */
// TODO: Does this need to be a class? Or should we just define a type 
// DiagonalOperator which is an std::vector<nda::array<dcomplex,3>>?
class DiagonalOperator {
    private: 
        std::vector<nda::array<dcomplex,3>> blocks;
        int num_block_rows;

    public:
        void set_blocks(std::vector<nda::array<dcomplex,3>> &blocks);
        std::vector<nda::array<dcomplex,3>> get_blocks() const;
        int get_num_block_rows() const;

    /**
     * @brief Constructor for DiagonalOperator
     * @param[in] blocks vector of diagonal blocks
     */
    DiagonalOperator(std::vector<nda::array<dcomplex,3>> &blocks);
};

/**
 * @class FOperator
 * @brief Block-sparse storage of F or F^dagger
 */
class FOperator {
    private: 
        nda::vector<int> block_indices;
        std::vector<nda::array<dcomplex,2>> blocks;
        int num_block_rows;
        int num_blocks; // number of block indices that are not -1

    public:
        void set_block_indices(nda::vector<int> &block_indices);
        void set_blocks(std::vector<nda::array<dcomplex,2>> &blocks);
        nda::vector<int> get_block_indices() const;
        std::vector<nda::array<dcomplex,2>> get_blocks() const;
        int get_num_block_rows() const;
        int get_num_blocks() const;

    /**
     * @brief Constructor for FOperator
     * @param[in] block_indices vector of block-row-indices
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in row i
     */
    FOperator(nda::vector<int> &block_indices, 
        std::vector<nda::array<dcomplex,2>> &blocks);
};

/**
 * @class BlockOperator
 * @brief Block-sparse storage of an arbitrary time-dependent operator
 */
class BlockOperator {
    private: 
        nda::vector<int> block_indices;
        std::vector<nda::array<dcomplex,3>> blocks;
        int num_block_rows;
        int num_blocks; // number of indices that are not -1

    public:
        void set_block_indices(nda::vector<int> &block_indices);
        void set_blocks(std::vector<nda::array<dcomplex,3>> &blocks);
        nda::vector<int> get_block_indices() const;
        std::vector<nda::array<dcomplex,3>> get_blocks() const;
        int get_num_block_rows() const;
        int get_num_blocks() const;

    /**
     * @brief Constructor for BlockOperator
     * @param[in] block_indices vector of block-row-indices
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in row i
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
 */
FOperator dagger_bs(FOperator const &F);

/**
 * @brief Evaluate matrix products in NCA using block-sparse storage
 * @param[in] hyb hybridization function
 * @param[in] Gt Greens function
 * @param[in] F F operator
 * @param[in] F_dag F^dagger operator
 * @return NCA self-energy
 */
DiagonalOperator NCA_bs(nda::array_const_view<dcomplex,3> hyb, 
    const DiagonalOperator &Gt, 
    const std::vector<FOperator> &Fs);