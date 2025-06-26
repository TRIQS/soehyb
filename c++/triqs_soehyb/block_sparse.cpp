#include "block_sparse.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/utils.hpp>
#include <h5/format.hpp>
#include <iostream>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/print.hpp>
#include <ostream>
#include <string>
#include <vector>
#include <stdexcept>

using namespace nda;

/////////////// BlockDiagOpFun (BDOF) class ///////////////
BlockDiagOpFun::BlockDiagOpFun(
    std::vector<nda::array<dcomplex,3>> &blocks,
    nda::vector_const_view<int> zero_block_indices) : 
    blocks(blocks), num_block_cols(blocks.size()), zero_block_indices(zero_block_indices) {}


BlockDiagOpFun::BlockDiagOpFun(int r, 
    nda::vector_const_view<int> block_sizes) : 
    num_block_cols(block_sizes.size()) {

    std::vector<nda::array<dcomplex,3>> blocks(num_block_cols);
    zero_block_indices = nda::make_regular(-1*nda::ones<int>(num_block_cols));
    for (int i = 0; i < num_block_cols; i++) {
        blocks[i] = nda::zeros<dcomplex>(r, block_sizes[i], block_sizes[i]);
    }
    this->blocks = blocks;
}

void BlockDiagOpFun::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {

    this->blocks = blocks;
    num_block_cols = blocks.size();
    zero_block_indices = nda::zeros<int>(num_block_cols);
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

nda::vector<int> BlockDiagOpFun::get_block_sizes() const {
    nda::vector<int> block_sizes(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        block_sizes(i) = blocks[i].shape(1);
    }
    return block_sizes;
}

int BlockDiagOpFun::get_block_size(int i) const {
    return blocks[i].shape(1);
}

int BlockDiagOpFun::get_max_block_size() const {
    int max_block_size = 0; 
    for (int i = 0; i < num_block_cols; i++) {
        max_block_size = std::max(max_block_size, (int) blocks[i].extent(1)); 
    }
    return max_block_size; 
}

int BlockDiagOpFun::get_num_block_cols() const {
    return num_block_cols;
}

int BlockDiagOpFun::get_zero_block_index(int i) const {
    return zero_block_indices(i);
}

void BlockDiagOpFun::set_blocks_dlr_coeffs(imtime_ops &itops) {
    for (int i = 0; i < num_block_cols; i++) {
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

int BlockDiagOpFun::get_num_time_nodes() const {
    for (int i = 0; i < num_block_cols; i++) {
        if (zero_block_indices(i) != -1) {
            return blocks[i].shape(0);
        }
    }
    return 0; // BlockDiagOpFun is all zeros anyways
}

void BlockDiagOpFun::add_block(int i, nda::array_const_view<dcomplex,3> block) {
    blocks[i] = nda::make_regular(blocks[i] + block); // TODO: does this work?
}

std::string BlockDiagOpFun::hdf5_format() { return "BlockDiagOpFun"; }

void h5_write(h5::group g, const std::string& subgroup_name, const BlockDiagOpFun& BDOF) {
    auto sg = g.create_group(subgroup_name);
    h5::write_hdf5_format(sg, BDOF);
    for (int i = 0; i < BDOF.num_block_cols; i++) {
        h5::write(sg, "block_" + std::to_string(i), BDOF.blocks[i]);
    }
    h5::write(sg, "zero_block_indices", BDOF.zero_block_indices);
}

/////////////// BlockOp (BO) class ///////////////

BlockOp::BlockOp(
    nda::vector<int> &block_indices, 
    std::vector<nda::array<dcomplex,2>> &blocks) : 
    block_indices(block_indices), blocks(blocks), num_block_cols(block_indices.size()) {}

BlockOp::BlockOp(
    nda::vector_const_view<int> block_indices, nda::array_const_view<int,2> block_sizes) :
    block_indices(block_indices), num_block_cols(block_indices.size()) {

    std::vector<nda::array<dcomplex,2>> blocks(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        if (block_indices(i) != -1) {
            blocks[i] = nda::zeros<dcomplex>(block_sizes(i,0), block_sizes(i,1));
        }
        else {
            blocks[i] = nda::zeros<dcomplex>(1, 1);
        }
    }
    this->blocks = blocks;
}

BlockOp& BlockOp::operator+=(const BlockOp &F) {
    // BlockOp addition-assignment operator
    // @param[in] F BlockOp
    // TODO: exception handling
    for (int i = 0; i < this->num_block_cols; i++) {
        if (F.get_block_index(i) != -1) {
            this->blocks[i] += F.blocks[i];
        }
    }
    return *this;
}

void BlockOp::set_block_indices(
    nda::vector<int> &block_indices) {

    this->block_indices = block_indices;
    num_block_cols = block_indices.size();
}

void BlockOp::set_block_index(int i, int block_index) {
    block_indices(i) = block_index;
}

void BlockOp::set_blocks(
    std::vector<nda::array<dcomplex,2>> &blocks) {

    this->blocks = blocks;
    num_block_cols = blocks.size();
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
        auto arr = nda::zeros<dcomplex>(1,1);
        return arr;
    }
    else {
        return blocks[i];
    }
}

int BlockOp::get_num_block_cols() const { return num_block_cols; }

nda::array<int,2> BlockOp::get_block_sizes() const {
    auto block_sizes = nda::zeros<int>(num_block_cols,2);
    for (int i = 0; i < num_block_cols; i++) {
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

nda::vector<int> BlockOp::get_block_size(int i) const {
    auto block_size = nda::zeros<int>(2);
    if (block_indices(i) != -1) {
        block_size(0) = blocks[i].shape(0);
        block_size(1) = blocks[i].shape(1);
    }
    else {
        block_size() = -1;
    }
    return block_size;
};

int BlockOp::get_block_size(int block_ind, int dim) const {
    if (block_indices(block_ind) != -1) {
        return blocks[block_ind].shape(dim);
    }
    else {
        return -1;
    }
}

/////////////// BlockOpFun (BOF) class ///////////////

BlockOpFun::BlockOpFun(
    nda::vector_const_view<int> block_indices, 
    std::vector<nda::array<dcomplex,3>> &blocks) : 
    block_indices(block_indices), blocks(blocks), num_block_cols(block_indices.size()) {}

BlockOpFun::BlockOpFun(
    int r, 
    nda::vector_const_view<int> block_indices, 
    nda::array_const_view<int,2> block_sizes) :
    block_indices(block_indices), num_block_cols(block_indices.size()) {

    for (int i = 0; i < num_block_cols; i++) {
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
    num_block_cols = block_indices.size();
}

void BlockOpFun::set_block_index(int i, int block_index) {
    block_indices(i) = block_index;
}

void BlockOpFun::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {

    this->blocks = blocks;
    num_block_cols = blocks.size();
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
        auto arr = nda::zeros<dcomplex>(1, 1, 1);
        return arr;
    }
    else {
        return blocks[i];
    }
}

int BlockOpFun::get_num_block_cols() const {
    return num_block_cols;
}

nda::array<int,2> BlockOpFun::get_block_sizes() const {
    auto block_sizes = nda::zeros<int>(num_block_cols,2);
    for (int i = 0; i < num_block_cols; i++) {
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

nda::vector<int> BlockOpFun::get_block_size(int i) const {
    auto block_size = nda::zeros<int>(2);
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
    for (int i = 0; i < num_block_cols; i++) {
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

int BlockOpFun::get_num_time_nodes() const {
    for (int i = 0; i < num_block_cols; i++) {
        if (block_indices(i) != -1) {
            return blocks[i].shape(0);
        }
    }
    return 0; // BlockDiagOpFun is all zeros anyways
}

/////////////// DenseFSet class ///////////////
DenseFSet::DenseFSet(nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags, 
    nda::array_const_view<dcomplex, 3> hyb_coeffs, 
    nda::array_const_view<dcomplex, 3> hyb_refl_coeffs
) : Fs(Fs), F_dags(F_dags) {

    int n = Fs.extent(0); 
    int N = Fs.extent(1); 
    int r = hyb_coeffs.extent(0); 
    F_dag_bars = nda::array<dcomplex,4>(n,r,N,N);
    F_bars_refl = nda::array<dcomplex,4>(n,r,N,N);
    for (int lam = 0; lam < n; lam++) {
        for (int l = 0; l < r; l++) {
            for (int nu = 0; nu < n; nu++) {
                F_dag_bars(lam,l,_,_) += hyb_coeffs(l,nu,lam)*F_dags(nu,_,_);
                F_bars_refl(nu,l,_,_) += hyb_refl_coeffs(l,nu,lam)*Fs(lam,_,_);
            }
        }
    }
}

/////////////// Utilities and operator overrides ///////////////

std::ostream& operator<<(std::ostream& os, BlockDiagOpFun &D) {
    // Print BlockDiagOpFun
    // @param[in] os output stream
    // @param[in] D BlockDiagOpFun
    // @return output stream

    for (int i = 0; i < D.get_num_block_cols(); i++) {
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
    for (int i = 0; i < F.get_num_block_cols(); i++) {
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

    int num_block_cols = F.get_num_block_cols();
    int i, j;

    // find block indices for F^dagger
    nda::vector<int> block_indices_dag(num_block_cols);
    // initialize indices with -1
    block_indices_dag = -1;
    std::vector<nda::array<dcomplex,2>> blocks_dag(num_block_cols);
    for (i = 0; i < num_block_cols; ++i) {
        j = F.get_block_indices()[i];
        if (j != -1) {
            block_indices_dag[j] = i;
            blocks_dag[j] = nda::transpose(F.get_blocks()[i]);
        }
    }
    BlockOp F_dag(block_indices_dag, blocks_dag);
    return F_dag;
}

BlockOp operator*(const dcomplex c, const BlockOp &F) {    
    // Compute a product between a scalar and an BlockOp
    // @param[in] c dcomplex
    // @param[in] F BlockOp

    auto product = F;
    for (int i = 0; i < F.get_num_block_cols(); i++) {
        if (F.get_block_index(i) != -1) {
            auto prod_block = nda::make_regular(c*F.get_block(i));
            product.set_block(i, prod_block);
        }
    }
    return product;
}

BlockDiagOpFun BOFtoBDOF(BlockOpFun const &A) {
    // Convert a BlockOpFun with diagonal structure to a BlockDiagOpFun
    // @param[in] A BlockOpFun
    // @return BlockDiagOpFun

    int num_block_cols = A.get_num_block_cols();
    auto diag_blocks = A.get_blocks();
    auto zero_block_indices = nda::zeros<int>(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
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

BlockDiagOpFun nonint_gf_BDOF(std::vector<nda::array<double,2>> H_blocks, 
    nda::vector<int> H_block_inds, 
    double beta, 
    nda::vector_const_view<double> dlr_it) {

    int num_block_cols = H_block_inds.size();
    nda::vector<int> H_block_sizes(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        H_block_sizes(i) = H_blocks[i].extent(0);
    }
    
    int r = dlr_it.size();
    
    double tr_exp_minusbetaH = 0;
    std::vector<nda::array<double,1>> H_evals(num_block_cols);
    std::vector<nda::array<double,2>> H_evecs(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        if (H_block_inds(i) != -1) {
            if (H_block_sizes(i) == 1) {
                H_evals[i] = nda::array<double,1>{H_blocks[i](0,0)};
                H_evecs[i] = nda::array<double,2>{{1}};
            } else {
                auto H_block_eig = nda::linalg::eigenelements(H_blocks[i]);
                H_evals[i] = std::get<0>(H_block_eig);
                H_evecs[i] = std::get<1>(H_block_eig);
            }
            tr_exp_minusbetaH += nda::sum(exp(-beta*H_evals[i]));
        }
        else {
            H_evals[i] = nda::zeros<double>(H_block_sizes(i));
            H_evecs[i] = nda::eye<double>(H_block_sizes(i));
            tr_exp_minusbetaH += 1.0*H_block_sizes(i); // 0 entry in the diagonal
        }
    }

    auto eta_0 = nda::log(tr_exp_minusbetaH) / beta;
    auto Gt = BlockDiagOpFun(r, H_block_sizes);
    for (int i = 0; i < num_block_cols; i++) {
        auto Gt_block = nda::array<dcomplex,3>(r, H_block_sizes(i), H_block_sizes(i));
        auto Gt_temp = nda::make_regular(0*H_blocks[i]);
        for (int t = 0; t < r; t++) {
            for (int j = 0; j < H_block_sizes(i); j++) {
                Gt_temp(j,j) = -exp(-beta*dlr_it(t)*(H_evals[i](j) + eta_0));
            }
            Gt_block(t,_,_) = nda::matmul(
                H_evecs[i], 
                nda::matmul(Gt_temp, nda::transpose(H_evecs[i])));
        }
        Gt.set_block(i, Gt_block);
    }

    return Gt;
}
