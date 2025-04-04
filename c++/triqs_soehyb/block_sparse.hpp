#pragma once
#include "nda/nda.hpp"
#include "strong_cpl.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <vector>

using namespace nda;

// TODO:
// templates for double/dcomplex
// convolve on values, not DLR coeffs
// OCA testing: Delta, G = single exponentials. fermionic dimer, G = G0 = e^(i*H_dimer*t)

/**
 * @class BlockDiagOpFun
 * @brief Block-sparse storage of time-dependent block-diagonal operator (e.g. Green's f'n)
 */
// TODO: Does this need to be a class? Or should we just define a type 
// BlockDiagOpFun which is an std::vector<nda::array<dcomplex,3>>?
// Maybe yes -- class could also have DLR info?
class BlockDiagOpFun {
    private: 
        std::vector<nda::array<dcomplex,3>> blocks;
        int num_block_rows;
        nda::vector<int> zero_block_indices;
        std::vector<nda::array<dcomplex,3>> blocks_dlr_coeffs;

    public:
        BlockDiagOpFun& operator+=(const BlockDiagOpFun &G);
        void set_blocks(std::vector<nda::array<dcomplex,3>> &blocks);
        void set_block(int i, nda::array_const_view<dcomplex,3>);
        const std::vector<nda::array<dcomplex,3>>& get_blocks() const;
        nda::array_const_view<dcomplex,3> get_block(int i) const;
        nda::vector_const_view<int> get_block_sizes() const;
        const int get_block_size(int i) const;
        const int get_num_block_rows() const;
        const int get_zero_block_index(int i) const;
        void set_blocks_dlr_coeffs(imtime_ops &itops);
        const std::vector<nda::array<dcomplex,3>>& get_blocks_dlr_coeffs() const;
        nda::array_const_view<dcomplex,3> get_block_dlr_coeffs(int i) const;
        const int get_num_time_nodes() const;

    /**
     * @brief Constructor for BlockDiagOpFun
     * @param[in] blocks vector of diagonal blocks
     * @param[in] zero_block_indices if i-th entry is -1, then blocks(i) = 0
     */
    BlockDiagOpFun(std::vector<nda::array<dcomplex,3>> &blocks, 
        nda::vector_const_view<int> zero_block_indices);

    /**
     * @brief Constructor for BlockDiagOpFun with blocks of zeros
     * @param[in] r number of imaginary time nodes
     * @param[in] block_sizes vector of sizes of diagonal blocks
     */
    BlockDiagOpFun(int r, nda::vector_const_view<int> block_sizes);
};

/**
 * @class BlockOp
 * @brief Block-sparse storage of F or F^dagger
 */
class BlockOp {
    private: 
        nda::vector<int> block_indices;
        std::vector<nda::array<dcomplex,2>> blocks;
        int num_block_rows;

    public:
        BlockOp& operator+=(const BlockOp &F);
        void set_block_indices(nda::vector<int> &block_indices);
        void set_block_index(int i, int block_index);
        void set_blocks(std::vector<nda::array<dcomplex,2>> &blocks);
        void set_block(int i, nda::array_const_view<dcomplex,2>);
        nda::vector_const_view<int> get_block_indices() const;
        int get_block_index(int i) const;
        const std::vector<nda::array<dcomplex,2>>& get_blocks() const;
        nda::array_const_view<dcomplex,2> get_block(int i) const;
        const int get_num_block_rows() const;
        nda::array_const_view<int,2> get_block_sizes() const;
        nda::vector_const_view<int> get_block_size(int i) const;

    /**
     * @brief Constructor for BlockOp
     * @param[in] block_indices vector of block-column indices in each block-row
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in row i
     */
    BlockOp(nda::vector<int> &block_indices, 
        std::vector<nda::array<dcomplex,2>> &blocks);

    /**
     * @brief Constructor for BlockOp with blocks of zeros
     * @param[in] block_indices vector of block-column indices in each block-row
     * @param[in] block_sizes array of block sizes
     * @note block_indices[i] = -1 if F does not have a block in row i
     */
     BlockOp(nda::vector_const_view<int> block_indices, 
        nda::array_const_view<int,2> block_sizes);
};

/**
 * @class BlockOpFun
 * @brief Block-sparse storage of an arbitrary time-dependent operator
 */
class BlockOpFun {
    private: 
        nda::vector<int> block_indices;
        std::vector<nda::array<dcomplex,3>> blocks;
        int num_block_rows;
        std::vector<nda::array<dcomplex,3>> blocks_dlr_coeffs;

    public:
        BlockOpFun& operator+=(const BlockOpFun &A);
        void set_block_indices(nda::vector<int> &block_indices);
        void set_block_index(int i, int block_index);
        void set_blocks(std::vector<nda::array<dcomplex,3>> &blocks);
        void set_block(int i, nda::array_const_view<dcomplex,3>);
        nda::vector_const_view<int> get_block_indices() const;
        int get_block_index(int i) const;
        const std::vector<nda::array<dcomplex,3>>& get_blocks() const;
        nda::array_const_view<dcomplex,3> get_block(int i) const;
        const int get_num_block_rows() const;
        nda::array_const_view<int,2> get_block_sizes() const;
        nda::vector_const_view<int> get_block_size(int i) const;
        void set_blocks_dlr_coeffs(imtime_ops& itops);
        const std::vector<nda::array<dcomplex,3>>& get_blocks_dlr_coeffs();
        nda::array_const_view<dcomplex,3> get_block_dlr_coeffs(int i) const;
        const int get_num_time_nodes() const;

    /**
     * @brief Constructor for BlockOpFun 
     * @param[in] block_indices vector of block-row-indices
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in row i
     */
    BlockOpFun(nda::vector_const_view<int> block_indices, 
        std::vector<nda::array<dcomplex,3>> &blocks);

    /**
     * @brief Constructor for BlockOpFun with blocks of zeros
     * @param[in] r number of imaginary time nodes
     * @param[in] block_indices vector of block-row-indices
     * @param[in] block_sizes vector of sizes of diagonal blocks
     */
     BlockOpFun(int r, nda::vector_const_view<int> block_indices, 
        nda::array_const_view<int,2> block_sizes);
};

/**
 * @brief Print BlockDiagOpFun to output stream
 * @param[in] os output stream
 * @param[in] D BlockDiagOpFun
 */
std::ostream& operator<<(std::ostream& os, BlockDiagOpFun &D);

/**
 * @brief Print BlockOp to output stream
 * @param[in] os output stream
 * @param[in] F BlockOp
 */
std::ostream& operator<<(std::ostream& os, BlockOp &F);

/**
 * @brief Compute the adjoint of an BlockOp
 * @param[in] F BlockOp
 * @return F^dagger operator
 */
BlockOp dagger_bs(BlockOp const &F);

/**
 * @brief Convert a BlockOpFun with diagonal structure to a BlockDiagOpFun
 * @param[in] A BlockOpFun
 * @return BlockDiagOpFun
 */
BlockDiagOpFun BOFtoBDOF(BlockOpFun const &A);

/**
 * @brief Compute a product between a BlockDiagOpFun and a BlockOpFun
 * @param[in] A BlockDiagOpFun
 * @param[in] B BlockOpFun
 */
BlockOpFun operator*(const BlockDiagOpFun& A, const BlockOpFun& B);

/**
 * @brief Compute a product between a BlockOpFun and a BlockDiagOpFun
 * @param[in] A BlockOpFun
 * @param[in] B BlockDiagOpFun
 */
BlockOpFun operator*(const BlockOpFun& A, const BlockDiagOpFun& B);

/**
 * @brief Compute a product between a BlockDiagOpFun and an BlockOp
 * @param[in] A BlockDiagOpFun
 * @param[in] F BlockOp
 */
BlockOpFun operator*(const BlockDiagOpFun& A, const BlockOp& F);

 /**
  * @brief Compute a product between an BlockOp and a BlockDiagOpFun
  * @param[in] F BlockOp
  * @param[in] B BlockDiagOpFun
  */
BlockOpFun operator*(const BlockOp& F, const BlockDiagOpFun& B);

/**
 * @brief Compute a product between a BlockOpFun and an BlockOp
 * @param[in] A BlockOpFun
 * @param[in] F BlockOp
 */
BlockOpFun operator*(const BlockOpFun& A, const BlockOp& F);

 /**
  * @brief Compute a product between an BlockOp and a BlockOpFun
  * @param[in] F BlockOp
  * @param[in] B BlockOpFun
  */
BlockOpFun operator*(const BlockOp& F, const BlockOpFun& B);

/**
 * @brief Compute a product between a scalar and an BlockOp
 * @param[in] c dcomplex
 * @param[in] F BlockOp
 */
BlockOp operator*(const dcomplex c, const BlockOp &F);

/**
 * @brief Compute a product between a scalar f'n of time and a BlockOpFun
 * @param[in] f nda::vector_const_view<double,1>
 * @param[in] A BlockOpFun
 */
BlockOpFun operator*(nda::vector_const_view<double> f, const BlockOpFun& A);

/**
 * @brief Compute a product between a BlockOpFun and a scalar f'n of time
 * @param[in] A BlockOpFun
 * @param[in] f nda::array_const_view<double,1>
 */
BlockOpFun operator*(const BlockOpFun& A, nda::vector_const_view<double> f);

/**
 * @brief Compute a product between a scalar f'n of time and a BlockOpFun
 * @param[in] f nda::vector_const_view<dcomplex,1>
 * @param[in] A BlockOpFun
 */
BlockOpFun operator*(nda::vector_const_view<dcomplex> f, const BlockOpFun& A);

/**
 * @brief Compute a product between a BlockOpFun and a scalar f'n of time
 * @param[in] A BlockOpFun
 * @param[in] f nda::array_const_view<dcomplex,1>
 */
BlockOpFun operator*(const BlockOpFun& A, nda::vector_const_view<dcomplex> f);

/**
 * @brief Compute a quotient between a BlockDiagOpFun and a scalar
 * @param[in] A BlockDiagOpFun
 * @param[in] c dcomplex
 */
BlockDiagOpFun operator/(const BlockDiagOpFun& A, dcomplex c);

/**
 * @brief Convolve a BlockDiagOpFun and a BlockOpFun
 */
BlockOpFun convolve(
    imtime_ops itops,
    double beta,
    statistic_t statistic,
    const BlockDiagOpFun& f, 
    const BlockOpFun& g,
    bool time_order = false);

/**
 * @brief Convolve a BlockDiagOpFun and a BlockOpFun
 */
 BlockOpFun convolve(
    imtime_ops itops,
    double beta,
    statistic_t statistic,
    const BlockOpFun& f,
    const BlockDiagOpFun& g, 
    bool time_order = false);


/**
 * @brief Evaluate NCA using block-sparse storage
 * @param[in] hyb hybridization function
 * @param[in] Gt Greens function
 * @param[in] F F operator
 * @param[in] F_dag F^dagger operator
 * @return NCA term of self-energy
 */
BlockDiagOpFun NCA_bs(nda::array_const_view<dcomplex,3> hyb, 
    const BlockDiagOpFun &Gt, 
    const std::vector<BlockOp> &Fs);

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
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] hyb_coeffs DLR coefficients of hybridization
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] F F operator
 * @param[in] F_dag F^dagger operator
 * @return OCA term of self-energy
 */
BlockDiagOpFun OCA_bs(nda::array_const_view<dcomplex,3> hyb,
    nda::array_const_view<dcomplex,3> hyb_coeffs,
    imtime_ops &itops, 
    double beta, 
    const BlockDiagOpFun &Gt, 
    const std::vector<BlockOp> &Fs);