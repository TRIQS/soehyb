#include "block_sparse.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <iostream>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/print.hpp>
#include <vector>
#include <stdexcept>

using namespace nda;

BlockDiagOpFun::BlockDiagOpFun(
    std::vector<nda::array<dcomplex,3>> &blocks,
    nda::vector_const_view<int> zero_block_indices) : 
    blocks(blocks), zero_block_indices(zero_block_indices) {
        
    num_block_rows = blocks.size();
}

BlockDiagOpFun::BlockDiagOpFun(int r, 
    nda::vector_const_view<int> block_sizes) {

    num_block_rows = block_sizes.size();
    for (int i = 0; i < num_block_rows; i++) {
        blocks[i] = nda::zeros<dcomplex>(r, block_sizes[i], block_sizes[i]);
        zero_block_indices(i) = -1; 
    }
}

void BlockDiagOpFun::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {
        
    this->blocks = blocks;
    num_block_rows = blocks.size();
    zero_block_indices = nda::zeros<int>(num_block_rows);
}

void BlockDiagOpFun::set_block(int i, nda::array_const_view<dcomplex,3> block) {
    blocks[i] = block;
    zero_block_indices(i) = 0;
}

const std::vector<nda::array<dcomplex,3>>& BlockDiagOpFun::get_blocks() const {        
    return blocks;
}

nda::array_const_view<dcomplex,3> BlockDiagOpFun::get_block(int i) const {
    return blocks[i];
}

nda::vector_const_view<int> BlockDiagOpFun::get_block_sizes() const {
    static nda::vector<int> block_sizes(num_block_rows);
    for (int i = 0; i < num_block_rows; i++) {
        block_sizes(i) = blocks[i].shape(1);
    }
    return block_sizes;
}

const int BlockDiagOpFun::get_block_size(int i) const {
    return blocks[i].shape(1);
}

const int BlockDiagOpFun::get_num_block_rows() const {
    return num_block_rows;
}

const int BlockDiagOpFun::get_zero_block_index(int i) const {
    return zero_block_indices(i);
}

void BlockDiagOpFun::set_blocks_dlr_coeffs(imtime_ops &itops) {    
    for (int i = 0; i < num_block_rows; i++) {
        blocks_dlr_coeffs[i] = itops.vals2coefs(blocks[i]);
    }
}

const std::vector<nda::array<dcomplex,3>>& 
    BlockDiagOpFun::get_blocks_dlr_coeffs() const {

    return blocks_dlr_coeffs;
}

nda::array_const_view<dcomplex,3>
    BlockDiagOpFun::get_block_dlr_coeffs(int i) const {

    return blocks_dlr_coeffs[i];
}

const int BlockDiagOpFun::get_num_time_nodes() const {
    for (int i; i < num_block_rows; i++) {
        if (zero_block_indices(i) != -1) {
            return blocks[i].shape(0);
        }
    }
    return 0; // BlockDiagOpFun is all zeros anyways
}


BlockOp::BlockOp(
    nda::vector<int> &block_indices, 
    std::vector<nda::array<dcomplex,2>> &blocks) : 
    block_indices(block_indices), blocks(blocks) {

    num_block_rows = block_indices.size();
}

BlockOp::BlockOp(
    nda::vector_const_view<int> block_indices, nda::array_const_view<int,2> block_sizes) :
    block_indices(block_indices) {

    num_block_rows = block_indices.size();
    for (int i = 0; i < num_block_rows; i++) {
        if (block_indices(i) != -1) {
            blocks[i] = nda::zeros<dcomplex>(block_sizes(i,0), block_sizes(i,1));
        }
        else {
            blocks[i] = nda::zeros<dcomplex>(1, 1);
        }
    }
}

void BlockOp::set_block_indices(
    nda::vector<int> &block_indices) {

    this->block_indices = block_indices;
    num_block_rows = block_indices.size();
}

void BlockOp::set_blocks(
    std::vector<nda::array<dcomplex,2>> &blocks) {

    this->blocks = blocks;
    num_block_rows = blocks.size();
}

void BlockOp::set_block(int i, nda::array_const_view<dcomplex,2> block) {
    blocks[i] = block;
}

nda::vector_const_view<int> BlockOp::get_block_indices() const {
    return block_indices;
}

int BlockOp::get_block_index(int i) const { return block_indices(i); }

const std::vector<nda::array<dcomplex,2>>& BlockOp::get_blocks() const {
    return blocks;
}

nda::array_const_view<dcomplex,2> BlockOp::get_block(int i) const {
    if (block_indices(i) == -1) {
        static auto arr = nda::zeros<dcomplex>(1,1);
        return arr;
    }
    else {
        return blocks[i];
    }
}

const int BlockOp::get_num_block_rows() const { return num_block_rows; }

nda::array_const_view<int,2> BlockOp::get_block_sizes() const {
    static nda::array<int,2> block_sizes(num_block_rows,2);
    for (int i = 0; i < num_block_rows; i++) {
        if (block_indices(i) != -1) {
            block_sizes(i,0) = blocks[i].shape(0);
            block_sizes(i,1) = blocks[i].shape(1);
        }
        else {
            block_sizes(i,0) = -1;
            block_sizes(i,1) = -1;
        }
    }
    return block_sizes;
};
        
nda::vector_const_view<int> BlockOp::get_block_size(int i) const {
    static nda::vector<int> block_size(2);
    if (block_indices(i) != -1) {
        block_size(0) = blocks[i].shape(0);
        block_size(1) = blocks[i].shape(1);
    }
    else {
        block_size() = -1;
    }
    return block_size;
};


BlockOpFun::BlockOpFun(
    nda::vector_const_view<int> block_indices, 
    std::vector<nda::array<dcomplex,3>> &blocks) : 
    block_indices(block_indices), blocks(blocks) {
        
    num_block_rows = block_indices.size();
}

BlockOpFun::BlockOpFun(
    int r, 
    nda::vector_const_view<int> block_indices, 
    nda::array_const_view<int,2> block_sizes) :
    block_indices(block_indices) {

    num_block_rows = block_indices.size();
    for (int i = 0; i < num_block_rows; i++) {
        if (block_indices(i) != -1) {
            blocks[i] = nda::zeros<dcomplex>(r, block_sizes(i,0), block_sizes(i,1));
        }
        else {
            blocks[i] = nda::zeros<dcomplex>(1, 1, 1);
        }
    }
}

void BlockOpFun::set_block_indices(
    nda::vector<int> &block_indices) {

    this->block_indices = block_indices;
    num_block_rows = block_indices.size();
}

void BlockOpFun::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {

    this->blocks = blocks;
    num_block_rows = blocks.size();
}

void BlockOpFun::set_block(int i, nda::array_const_view<dcomplex,3> block) {
    blocks[i] = block;
}

nda::vector_const_view<int> BlockOpFun::get_block_indices() const {
    return block_indices;
}

int BlockOpFun::get_block_index(int i) const {
    return block_indices(i);
}

const std::vector<nda::array<dcomplex,3>>& BlockOpFun::get_blocks() const {
    return blocks;
}

nda::array_const_view<dcomplex,3> BlockOpFun::get_block(int i) const {
    if (block_indices(i) == -1) {
        static auto arr = nda::zeros<dcomplex>(1, 1, 1);
        return arr;
    }
    else {
        return blocks[i];
    }
}

const int BlockOpFun::get_num_block_rows() const {
    return num_block_rows;
}

nda::array_const_view<int,2> BlockOpFun::get_block_sizes() const {
    static nda::array<int,2> block_sizes(num_block_rows,2);
    for (int i = 0; i < num_block_rows; i++) {
        if (block_indices(i) != -1) {
            block_sizes(i,0) = blocks[i].shape(1);
            block_sizes(i,1) = blocks[i].shape(2);
        }
        else {
            block_sizes(i,0) = -1;
            block_sizes(i,1) = -1;
        }
    }
    return block_sizes;
}

nda::vector_const_view<int> BlockOpFun::get_block_size(int i) const {
    static nda::vector<int> block_size(2);
    if (block_indices(i) != -1) {
        block_size(0) = blocks[i].shape(1);
        block_size(1) = blocks[i].shape(2);
    }
    else {
        block_size() = -1;
    }
    return block_size;
}

void BlockOpFun::set_blocks_dlr_coeffs(imtime_ops &itops) {
    for (int i = 0; i < num_block_rows; i++) {
        if (block_indices(i) != -1) {
            blocks_dlr_coeffs[i] = itops.vals2coefs(blocks[i]);
        }
    }
}

const std::vector<nda::array<dcomplex,3>>& BlockOpFun::get_blocks_dlr_coeffs() {
    return blocks_dlr_coeffs;
}

nda::array_const_view<dcomplex,3> BlockOpFun::get_block_dlr_coeffs(int i) const {
    return blocks_dlr_coeffs[i];
}

const int BlockOpFun::get_num_time_nodes() const {
    for (int i; i < num_block_rows; i++) {
        if (block_indices(i) != -1) {
            return blocks[i].shape(0);
        }
    }
    return 0; // BlockDiagOpFun is all zeros anyways
}


std::ostream& operator<<(std::ostream& os, BlockDiagOpFun &D) {
    // Print BlockDiagOpFun
    // @param[in] os output stream
    // @param[in] D BlockDiagOpFun
    // @return output stream

    for (int i = 0; i < D.get_num_block_rows(); i++) {
        os << "Block " << i << ":\n" << D.get_block(i) << "\n";
    }
    return os;
};

std::ostream& operator<<(std::ostream& os, BlockOp &F) {
    // Print BlockOp
    // @param[in] os output stream
    // @param[in] F BlockOp
    // @return output stream

    os << "Block indices: " << F.get_block_indices() << "\n";
    for (int i = 0; i < F.get_num_block_rows(); i++) {
        if (F.get_block_indices()[i] == -1) {
            os << "Block " << i << ": 0\n";
        }
        else {
            os << "Block " << i << ":\n" << F.get_block(i) << "\n";
        }
    }
    return os;
};

BlockOp dagger_bs(BlockOp const &F) {
    // Evaluate F^dagger in block-sparse storage
    // @param[in] F F operator
    // @return F^dagger operator

    int num_block_rows = F.get_num_block_rows();
    int i, j;

    // find block indices for F^dagger
    nda::vector<int> block_indices_dag(num_block_rows);
    // initialize indices with -1
    block_indices_dag = -1;
    std::vector<nda::array<dcomplex,2>> blocks_dag(num_block_rows);
    for (i = 0; i < num_block_rows; ++i) {
        j = F.get_block_indices()[i];
        if (j != -1) {
            block_indices_dag[j] = i;
            blocks_dag[j] = nda::transpose(F.get_blocks()[i]);
        }
    }
    BlockOp F_dag(block_indices_dag, blocks_dag);
    return F_dag;
}

BlockDiagOpFun BOFtoBDOF(BlockOpFun const &A) {
    // Convert a BlockOpFun with diagonal structure to a BlockDiagOpFun
    // @param[in] A BlockOpFun
    // @return BlockDiagOpFun

    int num_block_rows = A.get_num_block_rows();
    auto diag_blocks = A.get_blocks();
    auto block_indices = A.get_block_indices();
    auto zero_block_indices = nda::zeros<int>(num_block_rows);
    for (int i = 0; i < num_block_rows; i++) {
        int block_index = A.get_block_index(i);
        if (block_index == -1) {
            diag_blocks[i] = nda::zeros<dcomplex>(1, 1, 1);
            zero_block_indices(i) = -1;
        }
        else if (block_index != i) {
            throw std::invalid_argument("BOF is not diagonal");
        }
    }

    return BlockDiagOpFun(diag_blocks, zero_block_indices);
}

BlockDiagOpFun& BlockDiagOpFun::operator+=(const BlockDiagOpFun &G) {
    // BlockDiagOpFun addition-assignment operator
    // @param[in] G BlockDiagOpFun
    
    // TODO: exception handling
    for (int i = 0; i < num_block_rows; i++) {
        if (this->zero_block_indices(i) == -1 && G.get_zero_block_index(i) != -1) {
            this->blocks[i] = G.blocks[i];
            this->zero_block_indices(i) = 0;
        }
        else if (zero_block_indices(i) != -1 && G.get_zero_block_index(i) != -1) {
            this->blocks[i] += G.blocks[i];
        }
    }
    return *this;
}

BlockOp& BlockOp::operator+=(const BlockOp &F) {
    // BlockOp addition-assignment operator
    // @param[in] F BlockOp
    
    // TODO: exception handling
    for (int i = 0; i < num_block_rows; i++) {
        this->blocks[i] += F.blocks[i];
    }
    return *this;
}

BlockOpFun& BlockOpFun::operator+=(const BlockOpFun &A) {
    // BlockOpFun addition-assignment operator
    // @param[in] A BlockOpFun
    
    // TODO: exception handling
    for (int i = 0; i < num_block_rows; i++) {
        this->blocks[i] += A.blocks[i];
    }
    return *this;
}

BlockOpFun operator*(
    const BlockDiagOpFun& A, 
    const BlockOpFun& B) {
    // Compute a product between a BlockDiagOpFun and a BlockOpFun
    // @param[in] A BlockDiagOpFun
    // @param[in] B BlockOpFun

    BlockOpFun product = B;
    int r = B.get_num_time_nodes();
    if (r != A.get_num_time_nodes()) {
        throw std::invalid_argument("number of time indices do not match");
    }
    for (int i = 0; i < A.get_num_block_rows(); i++) {
        if (A.get_zero_block_index(i) == -1 || B.get_block_index(i) == -1) {
            // block i of A is zero, or block-row i of B has no nonzero block
            product.set_block(i, nda::zeros<dcomplex>(1, 1, 1));
        }
        else {
            auto prod_block = B.get_block(i);
            for (int t = 0; t < r; t++) {
                prod_block(t,_,_) = nda::matmul(
                    A.get_block(i)(t,_,_), prod_block(t,_,_));
            }
            product.set_block(i, prod_block);
        }
    }

    return product;
}

BlockOpFun operator*(
    const BlockOpFun& A, 
    const BlockDiagOpFun& B) {
    // Compute a product between a BlockOpFun and a BlockDiagOpFun
    // @param[in] A BlockOpFun
    // @param[in] B BlockDiagOpFun

    // initialize blocks of product, which has same shape as B
    BlockOpFun product = A;
    int r = A.get_num_time_nodes();
    if (r != B.get_num_time_nodes()) {
        throw std::invalid_argument("number of time indices does not match");
    }
    for (int i = 0; i < B.get_num_block_rows(); i++) {
        int j = A.get_block_index(i);
        if (j == -1 || B.get_zero_block_index(i) == -1) {
            // block-row i of A has no nonzero block, or block j of B is zero
            product.set_block(i, nda::zeros<dcomplex>(1, 1, 1));
        }
        else {
            auto prod_block = A.get_block(i);
            for (int t = 0; t < r; t++) {
                prod_block(t,_,_) = nda::matmul(
                    A.get_block(i)(t,_,_), B.get_block(j)(t,_,_));
            }
            product.set_block(i, prod_block);
        }
    }

    return product;
}

BlockOpFun operator*(const BlockDiagOpFun& A, const BlockOp& F) {
    // Compute a product between a BlockDiagOpFun and an BlockOp
    // @param[in] A BlockDiagOpFun
    // @param[in] F BlockOp

    int r = A.get_num_time_nodes();
    BlockOpFun product(r, F.get_block_indices(), F.get_block_sizes());

    // compute blocks of product
    for (int i = 0; i < A.get_num_block_rows(); ++i) {
        if (A.get_zero_block_index(i) == -1 || F.get_block_index(i) == -1) {
            // if block i of A is zero, or block-row i of F has no nonzero block
            product.set_block(i, nda::zeros<dcomplex>(1, 1, 1));
        }
        else {
            auto prod_block = product.get_block(i);
            for (int t = 0; t < r; t++) {
                prod_block(t,_,_) = nda::matmul(
                    A.get_block(i)(t,_,_), F.get_block(i));
            }
            product.set_block(i, prod_block);
        }
    }

    return product;
}

BlockOpFun operator*(const BlockOp& F, const BlockDiagOpFun& B) {
    // Compute a product between an BlockOp and a BlockDiagOpFun
    // @param[in] F BlockOp
    // @param[in] B BlockDiagOpFun

    int r = B.get_num_time_nodes();
    BlockOpFun product(r, F.get_block_indices(), F.get_block_sizes());

    // compute blocks of product
    for (int i = 0; i < B.get_num_block_rows(); ++i) {
        if (F.get_block_index(i) == -1 || B.get_zero_block_index(i) == -1) {
            // if block-row i of F has no nonzero block, or block i of B is zero
            product.set_block(i, nda::zeros<dcomplex>(1, 1, 1));
        }
        else {
            auto prod_block = product.get_block(i);
            for (int t = 0; t < r; t++) {
                prod_block(t,_,_) = nda::matmul(
                    F.get_block(i), B.get_block(F.get_block_index(i))(t,_,_));
            }
        }
    }

    return product;
}

// TODO: define BlockOpFun times BlockOp

BlockOp operator*(const dcomplex c, const BlockOp &F) {    
    // Compute a product between a scalar and an BlockOp
    // @param[in] c dcomplex
    // @param[in] F BlockOp
 
    auto product = F;
    for (int i = 0; i < F.get_num_block_rows(); i++) {
        if (F.get_block_index(i) != -1) {
            auto prod_block = nda::make_regular(c*F.get_block(i));
            product.set_block(i, prod_block);
        }
    }
    
    return product;
}

// TODO: define real function times BlockOp

BlockOpFun operator*(
    nda::vector_const_view<dcomplex>& f, 
    const BlockOpFun& A) {
    // Compute a product between a scalar f'n of time and a BlockOpFun
    // @param[in] f nda::array_const_view<dcomplex,1>
    // @param[in] A BlockOpFun
    // @return product

    BlockOpFun product = A;
    int r = A.get_num_time_nodes();
    for (int i = 0; i < A.get_num_block_rows(); i++) {
        if (A.get_block_index(i) != -1) {
            auto prod_block = product.get_block(i);
            for (int t = 0; t < r; t++) {
                prod_block(t,_,_) = nda::make_regular(f(t)*A.get_block(i)(t,_,_)); \
            }
            product.set_block(i, prod_block);
        }
    }
    return product;
}

BlockOpFun operator*(
    const BlockOpFun& A,
    nda::vector_const_view<dcomplex>& f) {
    // Compute a product between a scalar f'n of time and a BlockOpFun
    // @param[in] A BlockOpFun
    // @param[in] f nda::array_const_view<dcomplex,1>
    // @return product

    BlockOpFun product = A;
    int r = A.get_num_time_nodes();
    for (int i = 0; i < A.get_num_block_rows(); i++) {
        if (A.get_block_index(i) != -1) {
            auto prod_block = product.get_block(i);
            for (int t = 0; t < r; t++) {
                prod_block(t,_,_) = nda::make_regular(f(t)*A.get_block(i)(t,_,_)); \
            }
            product.set_block(i, prod_block);
        }
    }
    return product;
}

BlockDiagOpFun operator/(const BlockDiagOpFun& A, dcomplex c) {
    // Compute a quotient between a BlockDiagOpFun and a scalar
    // @param[in] A BlockDiagOpFun
    // @param[in] c dcomplex

    BlockDiagOpFun quotient = A;
    for (int i = 0; i < A.get_num_block_rows(); i++) {
        auto block = A.get_block(i);
        block = block/c;
        quotient.set_block(i, block);
    }

    return quotient;
}

// TODO: implement this with values directly?
BlockOpFun convolve(
    imtime_ops itops, 
    double beta, 
    statistic_t statistic,
    const BlockDiagOpFun& f,
    const BlockOpFun& g,
    bool time_order) {

    BlockOpFun h = g;

    for (int i = 0; i < f.get_num_block_rows(); i++) {
        if (f.get_zero_block_index(i) == -1 || g.get_block_index(i) == -1) {
            h.set_block(i, nda::zeros<dcomplex>(1, 1, 1));
            h.set_block_index(i, -1);
        }
        else {
            auto h_block = itops.convolve(
                beta, 
                statistic, 
                f.get_block_dlr_coeffs(i), 
                g.get_block_dlr_coeffs(i), 
                time_order);
            h.set_block(i, h_block);
        }
    }
    
    return h;
}

BlockOpFun convolve(
    imtime_ops itops,
    double beta,
    statistic_t statistic,
    const BlockOpFun& f,
    const BlockDiagOpFun& g,
    bool time_order) {

    BlockOpFun h = f;

    for (int i = 0; i < f.get_num_block_rows(); i++) {
        int j = f.get_block_index(i);
        if (j == -1 || g.get_zero_block_index(j)) {
            h.set_block(i, nda::zeros<dcomplex>(1, 1, 1));
            h.set_block_index(i, -1);
        }
        else {
            auto h_block = itops.convolve(
                beta,
                statistic,
                f.get_block_dlr_coeffs(i),
                g.get_block_dlr_coeffs(j),
                time_order);
            h.set_block(i, h_block);
        }
    }

    return h;
}

// TODO: rewrite this
BlockDiagOpFun NCA_bs(
    nda::array_const_view<dcomplex,3> hyb, 
    BlockDiagOpFun const &Gt, 
    const std::vector<BlockOp> &Fs) {
    // Evaluate NCA using block-sparse storage
    // @param[in] hyb_self hybridization function
    // @param[in] Gt Greens function
    // @param[in] F_list F operators
    // @return NCA term of self-energy
    
    // get F^dagger operators
    int num_Fs = Fs.size();
    auto F_dags = Fs;
    for (int i = 0; i < num_Fs; ++i) {
        F_dags[i] = dagger_bs(Fs[i]);
    }
    // initialize blocks of self-energy, with same shape as Gt
    std::vector<nda::array<dcomplex,3>> diag_blocks(Gt.get_blocks());
    int num_block_rows = Gt.get_num_block_rows();
    for (int i = 0; i < num_block_rows; ++i) {
        diag_blocks[i] = 0; 
    }
    
    int r = Gt.get_num_time_nodes();
    BlockDiagOpFun Sigma(r, Gt.get_block_sizes());
    for (int t = 0; t < r; ++t) {
        // forward diagram contribution to self-energy
        // make loop over forward/backward for higher order diagrams 
        for (int l = 0; l < num_Fs; ++l) {
            BlockOp const &F_dag = F_dags[l]; 
            for (int k = 0; k < num_Fs; ++k) {
                BlockOp const &F = Fs[k];
                for (int i = 0; i < num_block_rows; ++i) {
                    int j = F_dag.get_block_index(i); // = col ind of block i
                    if (j != -1) { // if F_dag has block in row i
                        auto temp = nda::matmul(
                            F_dag.get_blocks()[i], Gt.get_blocks()[j](t,_,_));
                        auto prod_block = nda::matmul(
                            temp, F.get_blocks()[j]);
                        diag_blocks[i](t,_,_) += hyb(t,l,k)*prod_block;
                    }
                }
            }
        }
        // backward diagram contribution to self-energy
        for (int l = 0; l < num_Fs; ++l) {
            BlockOp const &F = Fs[l];
            for (int k = 0; k < num_Fs; ++k) {
                BlockOp const &F_dag = F_dags[k];
                for (int i = 0; i < num_block_rows; ++i) {
                    int j = F.get_block_indices()[i]; // = col ind of block i
                    if (j != -1) { // if F has block in row i
                        auto temp = nda::matmul(
                            F.get_block(i), Gt.get_block(j)(t,_,_));
                        auto prod_block = nda::matmul(
                            temp, F_dag.get_blocks()[j]);
                        diag_blocks[i](t,_,_) -= hyb(t,l,k)*prod_block;
                    }
                }
            }
        }
    }
    
    Sigma.set_blocks(diag_blocks);
    return Sigma;
}

 nda::array<double,2> K_mat(
    nda::vector_const_view<double> dlr_it,
    nda::vector_const_view<double> dlr_rf) {
    // @brief Build matrix of evaluations of K at imag times and real freqs
    // @param[in] dlr_it DLR imaginary time nodes
    // @param[in] dlr_rf DLR real frequencies
    // @return matrix of K evalutions

    int r = dlr_it.shape(0); // number of times = number of freqs
    nda::array<double,2> K(r, r);
    for (int k = 0; k < r; k++) {
        for (int l = 0; l < r; l++) {
            K(k,l) = k_it(dlr_it(k), dlr_rf(l));
        }
    }

    return K;
    }

BlockDiagOpFun OCA_bs(
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_coeffs, 
    imtime_ops &itops, 
    double beta, 
    const BlockDiagOpFun &Gt, 
    const std::vector<BlockOp> &Fs) {
    // Evaluate OCA using block-sparse storage
    // @param[in] hyb_coeffs DLR coefficients of hybridization
    // @param[in] itops cppdlr imaginary time object
    // @param[in] beta inverse temperature
    // @param[in] Gt Greens function at times dlr_it with DLR coefficients
    // @param[in] F F operator
    // @param[in] F_dag F^dagger operator
    // @return OCA term of self-energy

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.shape(0);

    // get F^dagger operators
    int num_Fs = Fs.size();
    auto F_dags = Fs;
    for (int i = 0; i < num_Fs; ++i) {
        F_dags[i] = dagger_bs(Fs[i]);
    }

    // evaluate matrices with (k,l)-entry K(tau_k,+-omega_l)
    nda::array<double,2> Kplus = K_mat(dlr_it, dlr_rf);
    nda::array<double,2> Kminus = K_mat(dlr_it, nda::make_regular(-dlr_rf));

    // compute Fbars and Fdagbars
    auto Fbar_indices = Fs[0].get_block_indices();
    auto Fbar_sizes = Fs[0].get_block_sizes();
    auto Fdagbar_indices = F_dags[0].get_block_indices();
    auto Fdagbar_sizes = F_dags[0].get_block_sizes(); 
    std::vector<std::vector<BlockOp>> Fbars(
        num_Fs, 
        std::vector<BlockOp>(
            num_Fs, 
            BlockOp(Fbar_indices, Fbar_sizes)));
    std::vector<std::vector<BlockOp>> Fdagbars(
        num_Fs, 
        std::vector<BlockOp>(
            num_Fs, 
            BlockOp(Fdagbar_indices, Fdagbar_sizes)));
    for (int lam = 0; lam < num_Fs; lam++) {
        for (int l = 0; l < r; l++) {
            for (int nu = 0; nu < num_Fs; nu++) {
                Fbars[lam][l] = hyb_coeffs(nu,lam,l)*Fs[nu];
                Fdagbars[lam][l] = hyb_coeffs(nu,lam,l)*F_dags[nu];
            }
        }
    }

    // initialize self-energy
    nda::vector<int> block_sizes = Gt.get_block_sizes();
    BlockDiagOpFun Sigma = BlockDiagOpFun(r, block_sizes);

    // loop over hybridization lines
    for (int fb1 = 1; fb1 > -1; fb1--) {
        for (int fb2 = 1; fb2 > -1; fb2--) {
            // fb = 1 for forward line, else = 0
            // fb1 corresponds with line from 0 to tau_2
            auto const &F1list = (fb1) ? Fs : F_dags;
            auto const &F2list = (fb2) ? Fs : F_dags;
            auto const &F3list = (fb1) ? F_dags : Fs;
            // std::vector<BlockOp> const &F4list = (fb2) ? F_dags : Fs;
            auto const Fbar_array = (fb2) ? Fdagbars : Fbars;
            int sfM = (fb1^fb2) ? 1 : -1; // sign

            for (int lam = 0; lam < num_Fs; lam++) {
                for (int l = 0; l < r; l++) {
                    bool omega_l_is_pos = (dlr_rf(l) > 0);

                    BlockOp const &F2 = F2list[lam]; 
                    BlockOpFun T(r, F2.get_block_indices(), F2.get_block_sizes());
                    if (omega_l_is_pos) {
                        // 1. multiply G(tau_2-tau_1) K^+(tau_2-tau_1) F_lambda
                        auto T = Kplus(_,l)*(Gt*F2);
                        // 2. convolve by G
                        T.set_blocks_dlr_coeffs(itops);
                        T = convolve(itops, beta, Fermion, T, Gt, TIME_ORDERED);
                    }
                    else {
                        // 1. multiply F_lambda G(tau_1) K^-(tau_1)
                        auto T = Kminus(_,l)*(F2*Gt); 
                        // 2. convolve by G
                        T.set_blocks_dlr_coeffs(itops);
                        T = convolve(itops, beta, Fermion, Gt, T, TIME_ORDERED);
                    }
                    // 3. for each kappa, multiply by F_kappa from right
                    std::vector<BlockOpFun> Tkap(num_Fs, T);
                    for (int kap = 0; kap < num_Fs; ++kap) {
                        BlockOp const &F1 = F1list[kap];
                        Tkap[kap] = T*F1;
                    }
                    nda::vector<int> Tkap_indices = Tkap[0].get_block_indices();
                    nda::array<int,2> Tkap_sizes = Tkap[0].get_block_sizes();
                    // 4. for each mu, kap, mult by Delta_mu_kap and sum kap
                    // std::vector<std::vector<BlockOpFun>> Tmukap(num_Fs, Tkap);
                    std::vector<BlockOpFun> Tmu(num_Fs, BlockOpFun(r, Tkap_indices, Tkap_sizes));
                    for (int mu = 0; mu < num_Fs; ++mu) {
                        for (int kap = 0; kap < num_Fs; ++kap) {
                            Tmu[mu] += hyb(_,mu,kap)*Tkap[kap];
                        }
                    }
                    T = BlockOpFun(r, Tkap_indices, Tkap_sizes);
                    // 5. multiply by F^dag_mu and sum over mu
                    for (int mu = 0; mu < num_Fs; mu++) {
                        BlockOp const &F3 = F3list[mu];
                        T += F3*Tmu[mu];
                    }
                    // 6. convolve by G
                    T = convolve(itops, beta, Fermion, Gt, T, TIME_ORDERED);
                    // 7. multiply by K^+(tau) * Fbar
                    T = Kplus(_,l)*(Fbar_array[lam][l]*T);
                    Sigma += BOFtoBDOF(T)/k_it(0, -dlr_rf(l));
                }
            }
        }
    }

    return Sigma;
}