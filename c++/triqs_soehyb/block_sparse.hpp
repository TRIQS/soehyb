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
        int num_block_cols;
        nda::vector<int> zero_block_indices;
        std::vector<nda::array<dcomplex,3>> blocks_dlr_coeffs;

    public:
        BlockDiagOpFun& operator+=(const BlockDiagOpFun &G);
        void set_blocks(std::vector<nda::array<dcomplex,3>> &blocks);
        void set_block(int i, nda::array_const_view<dcomplex,3> block);
        const std::vector<nda::array<dcomplex,3>>& get_blocks() const;
        nda::array_const_view<dcomplex,3> get_block(int i) const;
        nda::vector<int> get_block_sizes() const;
        int get_block_size(int i) const;
        int get_num_block_cols() const;
        int get_zero_block_index(int i) const;
        void set_blocks_dlr_coeffs(imtime_ops &itops);
        const std::vector<nda::array<dcomplex,3>>& get_blocks_dlr_coeffs() const;
        nda::array_const_view<dcomplex,3> get_block_dlr_coeffs(int i) const;
        int get_num_time_nodes() const;
        void add_block(int i, nda::array_const_view<dcomplex,3> block);
        static std::string hdf5_format();
        friend void h5_write(h5::group g, const std::string& subgroup_name, const BlockDiagOpFun& BDOF);
        friend void h5_read(h5::group g, const std::string& subgroup_name, BlockDiagOpFun& BDOF);

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
        int num_block_cols;

    public:
        BlockOp& operator+=(const BlockOp &F);
        void set_block_indices(nda::vector<int> &block_indices);
        void set_block_index(int i, int block_index);
        void set_blocks(std::vector<nda::array<dcomplex,2>> &blocks);
        void set_block(int i, nda::array_const_view<dcomplex,2> block);
        nda::vector_const_view<int> get_block_indices() const;
        int get_block_index(int i) const;
        const std::vector<nda::array<dcomplex,2>>& get_blocks() const;
        nda::array_const_view<dcomplex,2> get_block(int i) const;
        int get_num_block_cols() const;
        nda::array<int,2> get_block_sizes() const;
        nda::vector<int> get_block_size(int i) const;
        int get_block_size(int block_ind, int dim) const;

    /**
     * @brief Constructor for BlockOp
     * @param[in] block_indices vector of block-column indices in each block-col
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in col i
     */
    BlockOp(nda::vector<int> &block_indices, 
        std::vector<nda::array<dcomplex,2>> &blocks);

    /**
     * @brief Constructor for BlockOp with blocks of zeros
     * @param[in] block_indices vector of block-column indices in each block-col
     * @param[in] block_sizes array of block sizes
     * @note block_indices[i] = -1 if F does not have a block in col i
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
        int num_block_cols;
        std::vector<nda::array<dcomplex,3>> blocks_dlr_coeffs;

    public:
        BlockOpFun& operator+=(const BlockOpFun &A);
        void set_block_indices(nda::vector<int> &block_indices);
        void set_block_index(int i, int block_index);
        void set_blocks(std::vector<nda::array<dcomplex,3>> &blocks);
        void set_block(int i, nda::array_const_view<dcomplex,3> block);
        nda::vector_const_view<int> get_block_indices() const;
        int get_block_index(int i) const;
        const std::vector<nda::array<dcomplex,3>>& get_blocks() const;
        nda::array_const_view<dcomplex,3> get_block(int i) const;
        int get_num_block_cols() const;
        nda::array<int,2> get_block_sizes() const;
        nda::vector<int> get_block_size(int i) const;
        void set_blocks_dlr_coeffs(imtime_ops& itops);
        const std::vector<nda::array<dcomplex,3>>& get_blocks_dlr_coeffs();
        nda::array_const_view<dcomplex,3> get_block_dlr_coeffs(int i) const;
        int get_num_time_nodes() const;

    /**
     * @brief Constructor for BlockOpFun 
     * @param[in] block_indices vector of block-col-indices
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in col i
     */
    BlockOpFun(nda::vector_const_view<int> block_indices, 
        std::vector<nda::array<dcomplex,3>> &blocks);

    /**
     * @brief Constructor for BlockOpFun with blocks of zeros
     * @param[in] r number of imaginary time nodes
     * @param[in] block_indices vector of block-col-indices
     * @param[in] block_sizes vector of sizes of diagonal blocks
     */
     BlockOpFun(int r, nda::vector_const_view<int> block_indices, 
        nda::array_const_view<int,2> block_sizes);
};

/**
 * @class DenseFSet
 * @brief Container for (linear combinations of) creation and annihilation operators
 */
class DenseFSet {
    public:
        nda::array<dcomplex,3> Fs; 
        nda::array<dcomplex,3> F_dags; 
        nda::array<dcomplex,4> F_dag_bars; 
        nda::array<dcomplex,4> F_bars_refl;

    /**
     * @brief Constructor for DenseFSet
     * @param[in] Fs annihilation operators
     * @param[in] F_dags creation operators
     * @param[in] hyb_coeffs DLR coefficients of hybridization
     * @param[in] hyb_refl_coeffs DLR coefficients of reflected hybridization
     */
    DenseFSet(nda::array_const_view<dcomplex, 3> Fs, 
        nda::array_const_view<dcomplex, 3> F_dags, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> hyb_refl_coeffs); 
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
BlockDiagOpFun nonint_gf_BDOF(
    std::vector<nda::array<double,2>> H_blocks, 
    nda::vector<int> H_block_inds, 
    double beta, 
    nda::vector_const_view<double> dlr_it);

/**
 * @brief Compute noninteracting Green's function from Hamiltonian as a BDOF
 * @param[in] H_blocks vector of blocks of Hamiltonian
 * @param[in] H_block_inds vector, -1 if Hamiltonian has zero block in corresponding block column
 * @param[in] beta inverse temperature
 * @param[in] dlr_it imaginary time nodes in absolute format
 */
BlockDiagOpFun nonint_gf_BDOF(
    std::vector<nda::array<double,2>> H_blocks, 
    nda::vector<int> H_block_inds, 
    double beta, 
    nda::vector_const_view<double> dlr_it);
/**
 * @brief Compute noninteracting Green's function from Hamiltonian as a BDOF
 * @param[in] H_blocks vector of blocks of Hamiltonian
 * @param[in] H_block_inds vector, -1 if Hamiltonian has zero block in corresponding block column
 * @param[in] beta inverse temperature
 * @param[in] dlr_it imaginary time nodes in absolute format
 */
BlockDiagOpFun nonint_gf_BDOF(
    std::vector<nda::array<double,2>> H_blocks, 
    nda::vector<int> H_block_inds, 
    double beta, 
    nda::vector_const_view<double> dlr_it);
