#pragma once
#include "nda/nda.hpp"
#include "strong_cpl.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <vector>

using namespace nda;

/**
 * @class BlockDiagonalOperator
 * @brief Block-sparse storage of block-diagonal operator (e.g. Green's f'n)
 *
 * @note If block-row i has no block, blocks(i) MUST be 0. This is not necessary
 * @note for BlockOperators and FOperators
 */
// TODO: Does this need to be a class? Or should we just define a type 
// BlockDiagonalOperator which is an std::vector<nda::array<dcomplex,3>>?
// Maybe yes -- class could also have DLR info? TODO: add this
class BlockDiagonalOperator {
    private: 
        std::vector<nda::array<dcomplex,3>> blocks;
        int num_block_rows;
        vector<double> dlr_rf;
        vector<double> dlr_it;
        double Lambda;
        std::vector<nda::array<dcomplex,3>> block_dlr_coeffs;

    public:
        void set_blocks(std::vector<nda::array<dcomplex,3>> &blocks);
        const std::vector<nda::array<dcomplex,3>>& get_blocks() const;
        const nda::array<dcomplex,3>& get_block(int i) const;
        const int get_num_block_rows() const;

    /**
     * @brief Constructor for BlockDiagonalOperator
     * @param[in] blocks vector of diagonal blocks
     */
    BlockDiagonalOperator(std::vector<nda::array<dcomplex,3>> &blocks);
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
        const nda::vector<int>& get_block_indices() const;
        int get_block_index(int i) const;
        const std::vector<nda::array<dcomplex,2>>& get_blocks() const;
        const nda::array<dcomplex,2>& get_block(int i) const;
        const int get_num_block_rows() const;
        const int get_num_blocks() const;

    /**
     * @brief Constructor for FOperator
     * @param[in] block_indices vector of block-column indices in each block-row
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
        const nda::vector<int>& get_block_indices() const;
        int get_block_index(int i) const;
        const std::vector<nda::array<dcomplex,3>>& get_blocks() const;
        const nda::array<dcomplex,3>& get_block(int i) const;
        const int get_num_block_rows() const;
        const int get_num_blocks() const;

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
 * @brief Print BlockDiagonalOperator to output stream
 * @param[in] os output stream
 * @param[in] D BlockDiagonalOperator
 */
std::ostream& operator<<(std::ostream& os, BlockDiagonalOperator &D);

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
 * @brief Compute a product between a BlockDiagonalOperator and a BlockOperator
 * @param[in] A BlockDiagonalOperator
 * @param[in] B BlockOperator
 */
BlockOperator operator*(const BlockDiagonalOperator& A, const BlockOperator& B);

/**
 * @brief Compute a product between a BlockOperator and a BlockDiagonalOperator
 * @param[in] A BlockOperator
 * @param[in] B BlockDiagonalOperator
 */
BlockOperator operator*(const BlockOperator& A, const BlockDiagonalOperator& B);

/**
 * @brief Compute a product between a BlockDiagonalOperator and an FOperator
 * @param[in] A BlockDiagonalOperator
 * @param[in] F FOperator
 */
BlockOperator operator*(const BlockDiagonalOperator& A, const FOperator& F);

 /**
  * @brief Compute a product between a BlockOperator and a BlockDiagonalOperator
  * @param[in] F FOperator
  * @param[in] B BlockDiagonalOperator
  */
BlockOperator operator*(const FOperator& F, const BlockDiagonalOperator& B);

/**
 * @brief Compute a product between a scalar f'n of time and a BlockOperator
 * @param[in] f nda::vector_const_view<double,1>
 * @param[in] A BlockOperator
 */
BlockOperator operator*(nda::vector_const_view<double>& f, 
    const BlockOperator& A);

/**
 * @brief Compute a product between a BlockOperator and a scalar f'n of time
 * @param[in] A BlockOperator
 * @param[in] f nda::array_const_view<double,1>
 */
 BlockOperator operator*(const BlockOperator& A, 
    nda::vector_const_view<double>& f);

/**
 * @brief Evaluate NCA using block-sparse storage
 * @param[in] hyb hybridization function
 * @param[in] Gt Greens function
 * @param[in] F F operator
 * @param[in] F_dag F^dagger operator
 * @return NCA term of self-energy
 */
BlockDiagonalOperator NCA_bs(nda::array_const_view<dcomplex,3> hyb, 
    const BlockDiagonalOperator &Gt, 
    const std::vector<FOperator> &Fs);

/**
 * @brief Build matrix of evaluations of K at imag times and real freqs
 * @param[in] dlr_it DLR imaginary time nodes
 * @param[in] dlr_rf DLR real frequencies
 * @return matrix of K evalutions
 */
nda::array<double,2> K_mat(nda::vector_const_view<double> dlr_it,
    nda::vector_const_view<double> dlr_rf);

/**
 * @brief Evaluate OCA using block-sparse storage
 * @param[in] hyb_coeffs DLR coefficients of hybridization
 * @param[in] dlr_it DLR imaginary time nodes
 * @param[in] dlr_rf DLR real frequencies
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] F F operator
 * @param[in] F_dag F^dagger operator
 * @return OCA term of self-energy
 */
BlockDiagonalOperator OCA_bs(nda::array_const_view<dcomplex,3> hyb_coeffs,
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    imtime_ops &itops, 
    double beta, 
    const BlockDiagonalOperator &Gt, 
    const std::vector<FOperator> &Fs);