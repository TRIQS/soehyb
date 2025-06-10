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
        const int get_block_size(int i) const;
        const int get_num_block_cols() const;
        const int get_zero_block_index(int i) const;
        void set_blocks_dlr_coeffs(imtime_ops &itops);
        const std::vector<nda::array<dcomplex,3>>& get_blocks_dlr_coeffs() const;
        nda::array_const_view<dcomplex,3> get_block_dlr_coeffs(int i) const;
        const int get_num_time_nodes() const;
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
        const int get_num_block_cols() const;
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
        const int get_num_block_cols() const;
        nda::array<int,2> get_block_sizes() const;
        nda::vector<int> get_block_size(int i) const;
        void set_blocks_dlr_coeffs(imtime_ops& itops);
        const std::vector<nda::array<dcomplex,3>>& get_blocks_dlr_coeffs();
        nda::array_const_view<dcomplex,3> get_block_dlr_coeffs(int i) const;
        const int get_num_time_nodes() const;

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
 * @class BackboneTopology
 * @brief Representation of the common features of a given backbone diagram topology
 */
class BackboneTopology {
    private:
        std::vector<nda::vector<int>> vertices;
        std::vector<nda::vector<int>> edges;
        int vertex_to_0;

    /**
     * @brief Constructor for BackboneTopology
     * @param[in] hyb_edges nda::array_const_view array of hybridization edges
     */
    BackboneTopology(nda::array_const_view<int,2> hyb_edges);
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
 * @brief Evaluate NCA self-energy term using block-sparse storage
 * @param[in] hyb hybridization function
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] Gt Greens function
 * @param[in] Fs vector of annihilation operators
 * @return NCA term of self-energy
 */
BlockDiagOpFun NCA_bs(nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_refl, 
    const BlockDiagOpFun &Gt, 
    const std::vector<BlockOp> &Fs);

/**
 * @brief Evaluate NCA self-energy term using dense storage
 * @param[in] hyb hybridization function
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] Gt Greens function
 * @param[in] Fs vector of annihilation operators
 * @param[in] F_dags vector of creation operators
 */
nda::array<dcomplex,3> NCA_dense(
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_refl, 
    nda::array_const_view<dcomplex,3> Gt, 
    nda::array_const_view<dcomplex,3> Fs,
    nda::array_const_view<dcomplex,3> F_dags);

/**
 * @brief Build matrix of evaluations of K at imag times and real freqs
 * @param[in] dlr_it DLR imaginary time nodes
 * @param[in] dlr_rf DLR real frequencies
 * @return matrix of K evalutions
 */
nda::array<double,2> K_mat(nda::vector_const_view<double> dlr_it,
    nda::vector_const_view<double> dlr_rf, double beta);

/**
 * @brief DLR convolution routine for rectangular matrices
 * @param[in] itops CPPDLR imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] fc DLR coefficients of first function
 * @param[in] gc DLR coefficients of second function
 */
nda::array<dcomplex,3> convolve_rectangular(
    imtime_ops &itops, 
    double beta, 
    nda::array<dcomplex,3> fc, 
    nda::array<dcomplex,3> gc
);

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
 * @brief Evaluate OCA using block-sparse storage
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] Fs F operator
 * @return OCA term of self-energy
 */
BlockDiagOpFun OCA_bs(nda::array_const_view<dcomplex,3> hyb,
    imtime_ops &itops, 
    double beta, 
    const BlockDiagOpFun &Gt, 
    const std::vector<BlockOp> &Fs);

nda::array<dcomplex,3> eval_eq(imtime_ops &itops, nda::array_const_view<dcomplex, 3> f, int n_quad);

/**
 * @brief Evaluate OCA using dense storage
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] Fs F operator
 * @return OCA term of self-energy
 */
nda::array<dcomplex,3> OCA_dense(nda::array_const_view<dcomplex,3> hyb,
    imtime_ops &itops, 
    double beta, 
    nda::array_const_view<dcomplex,3> Gt, 
    nda::array_const_view<dcomplex,3> Fs, 
    nda::array_const_view<dcomplex,3> F_dags);

/**
 * @brief Evaluate OCA directly using trapezoidal quadrature
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] Fs F operator
 * @param[in] n_quad number of quadrature nodes
 * @return OCA term of self-energy
 */
nda::array<dcomplex,3> OCA_tpz(
    nda::array_const_view<dcomplex,3> hyb,
    imtime_ops &itops, 
    double beta, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    int n_quad);

/**
 * @brief Evaluate a single backbone diagram
 * @param[in] beta inverse temperature
 * @param[in] itops DLR imaginary time object
 * @param[in] vertex_0 BlockOp
 * @param[in] vertex_1 BlockOpFun
 * @param[in] right_vertices vector of BlockOps
 * @param[in] special_vertex BlockOpFun
 * @param[in] left_vertices vector of BlockOps
 * @param[in] edges vector of BlockDiagOpFuns
 * @param[in] prefactor 1-dimensional nda::array
 */
BlockDiagOpFun eval_backbone(double beta, 
    imtime_ops &itops, 
    const BlockOp &vertex_0, 
    const BlockOpFun &vertex_1, 
    const std::vector<BlockOp> &right_vertices, 
    const BlockOpFun &special_vertex, 
    const std::vector<BlockOp> &left_vertices, 
    const std::vector<BlockDiagOpFun> &edges, 
    nda::array_const_view<dcomplex,1> prefactor);