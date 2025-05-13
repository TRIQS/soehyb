#include "block_sparse.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <iostream>
#include <fstream>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/print.hpp>
#include <ostream>
#include <vector>
#include <stdexcept>

using namespace nda;

BlockDiagOpFun::BlockDiagOpFun(
    std::vector<nda::array<dcomplex,3>> &blocks,
    nda::vector_const_view<int> zero_block_indices) : 
    blocks(blocks), zero_block_indices(zero_block_indices) {
        
    num_block_cols = blocks.size();
}

BlockDiagOpFun::BlockDiagOpFun(int r, 
    nda::vector_const_view<int> block_sizes) {

    num_block_cols = block_sizes.size();
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

const int BlockDiagOpFun::get_block_size(int i) const {
    return blocks[i].shape(1);
}

const int BlockDiagOpFun::get_num_block_cols() const {
    return num_block_cols;
}

const int BlockDiagOpFun::get_zero_block_index(int i) const {
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

const int BlockDiagOpFun::get_num_time_nodes() const {
    for (int i; i < num_block_cols; i++) {
        if (zero_block_indices(i) != -1) {
            return blocks[i].shape(0);
        }
    }
    return 0; // BlockDiagOpFun is all zeros anyways
}

void BlockDiagOpFun::add_block(int i, nda::array_const_view<dcomplex,3> block) {
    blocks[i] = nda::make_regular(blocks[i] + block); // TODO: does this work?
}


BlockOp::BlockOp(
    nda::vector<int> &block_indices, 
    std::vector<nda::array<dcomplex,2>> &blocks) : 
    block_indices(block_indices), blocks(blocks) {

    num_block_cols = block_indices.size();
}

BlockOp::BlockOp(
    nda::vector_const_view<int> block_indices, nda::array_const_view<int,2> block_sizes) :
    block_indices(block_indices) {

    num_block_cols = block_indices.size();
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

const int BlockOp::get_num_block_cols() const { return num_block_cols; }

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


BlockOpFun::BlockOpFun(
    nda::vector_const_view<int> block_indices, 
    std::vector<nda::array<dcomplex,3>> &blocks) : 
    block_indices(block_indices), blocks(blocks) {
        
    num_block_cols = block_indices.size();
}

BlockOpFun::BlockOpFun(
    int r, 
    nda::vector_const_view<int> block_indices, 
    nda::array_const_view<int,2> block_sizes) :
    block_indices(block_indices) {

    num_block_cols = block_indices.size();
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

const int BlockOpFun::get_num_block_cols() const {
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

const int BlockOpFun::get_num_time_nodes() const {
    for (int i; i < num_block_cols; i++) {
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

void text2BlockOp(std::string fname) {
    // TODO docs

    std::ifstream bof1(fname);

    std::string line;
    int i = 0;
    int num_blocks;

    while (getline(bof1, line)) {
        // line 0 = number of blocks
        if (i == 0) {
            num_blocks = std::stoi(line);
        } 
        i += 1;
    }
    
    i = 0;
    std::ifstream bof2(fname);
    nda::array<int,2> block_sizes(num_blocks,2);
    nda::vector<int> block_indices(num_blocks);
    
    while (getline(bof2, line)) {
        // line 1 = block sizes
        if (i == 1) {
            int pos_start = 0;
            int pos_end;
            for (int j = 0; j < num_blocks; j++) {
                pos_end = line.find(',', pos_start);
                block_sizes(j,0) = std::stoi(line.substr(pos_start, pos_end - pos_start));
                pos_start = pos_end + 1;
                pos_end = line.find(',', pos_start);
                block_sizes(j,1) = std::stoi(line.substr(pos_start, pos_end - pos_start));
                pos_start = pos_end + 1;
            }
        }
        // line 2 = block indices
        else if (i == 2) {
            int pos_start = 0;
            int pos_end;
            for (int j = 0; j < num_blocks; j++) {
                pos_end = line.find(',', pos_start);
                block_indices(j) = std::stoi(line.substr(pos_start, pos_end - pos_start));
                pos_start = pos_end + 1;
            }
        }
        i += 1;
    }

    i = 0;
    std::ifstream bof3(fname);
    std::vector<nda::array<dcomplex,2>> blocks(num_blocks);

    while(getline(bof3, line)) {
        int curr_block_num = 0;
        if (i >= 3) {
            int pos_start = 0;
            int pos_end;
            if (line.substr(0,1) == "[") {
                nda::array<dcomplex,2> curr_block(
                    block_sizes(curr_block_num, 0), 
                    block_sizes(curr_block_num, 1));
                pos_end = line.find(' ', pos_start);
                std::cout << pos_end << std::endl;
            }
        }
        i += 1;
    }
}

BlockDiagOpFun BOFtoBDOF(BlockOpFun const &A) {
    // Convert a BlockOpFun with diagonal structure to a BlockDiagOpFun
    // @param[in] A BlockOpFun
    // @return BlockDiagOpFun

    int num_block_cols = A.get_num_block_cols();
    auto diag_blocks = A.get_blocks();
    auto block_indices = A.get_block_indices();
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

BlockDiagOpFun NCA_bs(
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_refl, 
    BlockDiagOpFun const &Gt, 
    const std::vector<BlockOp> &Fs) {
    // Evaluate NCA using block-sparse storage
    // @param[in] hyb hybridization function
    // @param[in] hyb_refl hybridization function eval'd at negative imag. times
    // @param[in] Gt Greens function
    // @param[in] F_list F operators
    // @return NCA term of self-energy
    
    // get F^dagger operators
    int num_Fs = Fs.size();
    auto F_dags = Fs;
    for (int i = 0; i < num_Fs; ++i) {
        F_dags[i] = dagger_bs(Fs[i]);
    }
    // initialize self-energy, with same shape as Gt
    int r = Gt.get_num_time_nodes();
    BlockDiagOpFun Sigma(r, Gt.get_block_sizes());

    for (int fb = 0; fb <= 1; fb++) {
        // fb = 1 for forward line, 0 for backward line
        auto const &F1list = (fb) ? Fs : F_dags;
        auto const &F2list = (fb) ? F_dags : Fs;
        int sfM = (fb) ? -1 : 1; 
        
        for (int lam = 0; lam < num_Fs; lam++) {
            for (int kap = 0; kap < num_Fs; kap++) {
                auto &F1 = F1list[kap];
                auto &F2 = F2list[lam];
                auto ind_path = nda::zeros<int>(2);
                bool path_all_nonzero; // if set to false during backwards pass,
                // the i-th block of Sigma is zero, so no computation needed

                for (int i = 0; i < Gt.get_num_block_cols(); i++) {
                    // "backwards pass"
                    // for each self-energy block, find contributing blocks of factors
                    path_all_nonzero = true; 
                    ind_path(0) = F1.get_block_index(i); // Sigma = F2 G F1
                    // ind_path(0) = block-column index of F1 corresponding with
                    // block that contributes to i-th block of Sigma
                    if (ind_path(0) != -1 && Gt.get_zero_block_index(ind_path(0)) != -1) {
                        ind_path(1) = F2.get_block_index(ind_path(0)); 
                        // ind_path(1) = block-column in F2 corresponding with
                        // block that contributes to i-th block of Sigma
                        //
                        // if F2's block is zero
                        if (ind_path(1) == -1) {path_all_nonzero = false;} 
                        
                    }
                    else { // F1's block or the block of Gt that would multiply this is zero
                        path_all_nonzero = false; 
                    }

                    // matmuls
                    // if path involves all nonzero blocks, compute product
                    // of blocks indexed by ind_path
                    if (path_all_nonzero) {
                        auto block = nda::zeros<dcomplex>(r, Gt.get_block_size(i), Gt.get_block_size(i));
                        for (int t = 0; t < r; t++) {
                            if (fb == 1) {
                                block(t,_,_) = hyb(t, lam, kap) * nda::matmul(
                                    F2.get_block(ind_path(0)), 
                                    nda::matmul(
                                        Gt.get_block(ind_path(0))(t,_,_), 
                                        F1.get_block(ind_path(1))));
                            } else {
                                block(t,_,_) = hyb_refl(t, lam, kap) * nda::matmul(
                                    F2.get_block(ind_path(0)), 
                                    nda::matmul(
                                        Gt.get_block(ind_path(0))(t,_,_), 
                                        F1.get_block(ind_path(1))));
                            }
                        }
                        block = sfM * block;
                        Sigma.add_block(i, block);
                    }
                }
            }
        }
    }
    
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

nda::array<dcomplex,3> convolve_rectangular(
    imtime_ops &itops, 
    double beta, 
    nda::array<dcomplex,3> f, 
    nda::array<dcomplex,3> g) {

    nda::array<dcomplex,3> h(f.extent(0), f.extent(1), g.extent(2));

    for (int i = 0; i < f.extent(1); i++) {
        for (int j = 0; j < g.extent(2); j++) {
            for (int k = 0; k < f.extent(2); k++) {
                h(_,i,j) += itops.convolve(beta, 
                    Fermion, 
                    itops.vals2coefs(f(_,i,k)), 
                    itops.vals2coefs(g(_,k,j)), 
                    TIME_ORDERED);
            }
        }
    }

    return h;
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
            for (int j = 0; j < H_block_sizes(i); j++) {
                tr_exp_minusbetaH += exp(-beta*H_evals[i](j));
            }
        }
        else {
            H_evals[i] = nda::zeros<double>(H_block_sizes(i));
            H_evecs[i] = nda::eye<double>(H_block_sizes(i));
            tr_exp_minusbetaH += 1.0*H_block_sizes(i); // 0 entry in the diagonal
        }
    }

    auto eta_0 = nda::log(tr_exp_minusbetaH) / beta;

    // TODO finish writing this function
    // create test combining call to this with beginning of twoband
    // start dedicated two_band test
    // check that Gt and H have the same zero block indices

    auto Gt = BlockDiagOpFun(r, H_block_sizes);
    for (int i = 0; i < num_block_cols; i++) {
        auto Gt_block = nda::array<dcomplex,3>(r, H_block_sizes(i), H_block_sizes(i));
        auto Gt_temp = nda::make_regular(0*H_blocks[i]);
        for (int t = 0; t < r; t++) {
            for (int j = 0; j < H_block_sizes(i); j++) {
                Gt_temp(j,j) = -exp(-dlr_it(t)*(H_evals[i](j) + eta_0));
            }
            Gt_block(t,_,_) = nda::matmul(
                H_evecs[i], 
                nda::matmul(Gt_temp, nda::transpose(H_evecs[i])));
        }
        Gt.set_block(i, Gt_block);
    }

    return Gt;
}

BlockDiagOpFun OCA_bs(
    nda::array_const_view<dcomplex,3> hyb,
    nda::array_const_view<dcomplex,3> hyb_refl,
    imtime_ops &itops, 
    double beta, 
    const BlockDiagOpFun &Gt, 
    const std::vector<BlockOp> &Fs) {
    // Evaluate OCA using block-sparse storage
    // @param[in] hyb hybridization on imaginary-time grid
    // @param[in] itops cppdlr imaginary time object
    // @param[in] beta inverse temperature
    // @param[in] Gt Greens function at times dlr_it with DLR coefficients
    // @param[in] Fs F operators
    // @return OCA term of self-energy

    // TODO: exceptions for bad argument sizes

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.shape(0);

    auto hyb_coeffs = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl); 

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
            r, 
            BlockOp(Fbar_indices, Fbar_sizes)));
    std::vector<std::vector<BlockOp>> Fdagbars(
        num_Fs, 
        std::vector<BlockOp>(
            r, 
            BlockOp(Fdagbar_indices, Fdagbar_sizes)));
    std::vector<std::vector<BlockOp>> Fbarsrefl(
        num_Fs, 
        std::vector<BlockOp>(
            r, 
            BlockOp(Fbar_indices, Fbar_sizes)));
    std::vector<std::vector<BlockOp>> Fdagbarsrefl(
        num_Fs, 
        std::vector<BlockOp>(
            r, 
            BlockOp(Fdagbar_indices, Fdagbar_sizes)));        
    for (int lam = 0; lam < num_Fs; lam++) {
        for (int l = 0; l < r; l++) {
            for (int nu = 0; nu < num_Fs; nu++) {
                Fbars[lam][l] += hyb_coeffs(l,nu,lam)*Fs[nu];
                Fdagbars[lam][l] += hyb_coeffs(l,nu,lam)*F_dags[nu];
                Fbarsrefl[lam][l] += hyb_refl_coeffs(l,nu,lam)*Fs[nu];
                Fdagbarsrefl[lam][l] += hyb_refl_coeffs(l,nu,lam)*F_dags[nu];
            }
        }
    }

    // initialize self-energy
    BlockDiagOpFun Sigma = BlockDiagOpFun(r, Gt.get_block_sizes());
    int num_block_cols = Gt.get_num_block_cols();

    // loop over hybridization lines
    for (int fb1 = 0; fb1 <= 1; fb1++) {
        for (int fb2 = 0; fb2 <= 1; fb2++) {
            // fb = 1 for forward line, else = 0
            // fb1 corresponds with line from 0 to tau_2
            auto const &F1list = (fb1) ? Fs : F_dags;
            auto const &F2list = (fb2) ? Fs : F_dags;
            auto const &F3list = (fb1) ? F_dags : Fs;
            auto const Fbar_array = (fb2) ? Fdagbars : Fbarsrefl;
            int sfM = (fb1^fb2) ? 1 : -1; // sign

            for (int i = 0; i < num_block_cols; i++) {

                // "backwards pass"
                // 
                // for each self-energy block, find contributing blocks of factors
                // 
                // paths_all_nonzero: false if for i-th block of 
                // Sigma, factors assoc'd with lambda, mu, kappa don't contribute
                // 
                // ind_path: a vector of column indices of the 
                // blocks of the factors that contribute. if paths_all_nonzero is 
                // false at this index, values in ind_path are garbage.
                //
                // ATTN: assumes all BlockOps in F(1,2,3)list have the same structure
                // i.e, index path is independent of kappa, mu
                bool path_all_nonzero = true;
                auto ind_path = nda::zeros<int>(3);
                auto Sigma_block_i = nda::make_regular(0*Sigma.get_block(i));

                int ip1 = F1list[0].get_block_index(i);
                ind_path(0) = ip1;
                if (ip1 == -1 || Gt.get_zero_block_index(ip1) == -1) {
                    path_all_nonzero = false;
                }
                else {
                    int ip2 = F2list[0].get_block_index(ip1);
                    ind_path(1) = ip2;
                    if (ip2 == -1 || Gt.get_zero_block_index(ip2) == -1) {
                        path_all_nonzero = false;
                    }
                    else {
                        int ip3 = F3list[0].get_block_index(ip2);
                        ind_path(2) = ip3;
                        if (ip3 == -1 
                            || Gt.get_zero_block_index(ip3) == -1 
                            || Fbar_array[0][0].get_block_index(ip3) == -1) {
                            path_all_nonzero = false;
                        }
                    }
                }

                // matmuls and convolutions
                if (path_all_nonzero) {
                    for (int l = 0; l < r; l++) {
                        bool omega_l_is_pos = (dlr_rf(l) > 0);
                        // initialize summand assoc'd with index l
                        auto Sigma_l = nda::make_regular(0*Sigma.get_block(i)); // TODO: preallocate?
                        for (int lam = 0; lam < num_Fs; lam++) {
                            auto F2_block = F2list[lam].get_block(ind_path(0));
                            auto T = nda::zeros<dcomplex>(
                                r,
                                F2_block.shape(0),
                                F2_block.shape(1));
                            if (omega_l_is_pos) {
                                // 1. multiply G(tau_2-tau_1) K^+(tau_2-tau_1) F_lambda
                                for (int t = 0; t < r; t++) {
                                    T(t,_,_) = Kplus(t,l) * nda::matmul(
                                        Gt.get_block(ind_path(1))(t,_,_),
                                        F2_block);
                                }
                                // 2. convolve by G
                                /*
                                T = itops.convolve(
                                    beta, 
                                    Fermion, 
                                    itops.vals2coefs(T), 
                                    itops.vals2coefs(Gt.get_block(ind_path(0))),
                                    TIME_ORDERED);*/
                                T = convolve_rectangular(itops, beta, T, Gt.get_block(ind_path(0)));
                            }
                            else {
                                // 1. multiply F_lambda G(tau_1) K^-(tau_1)
                                for (int t = 0; t < r; t++) {
                                    T(t,_,_) = Kminus(t,l) * nda::matmul(
                                        F2_block,
                                        Gt.get_block(ind_path(0))(t,_,_));
                                }
                                // 2. convolve by G
                                /*
                                T = itops.convolve(
                                    beta,
                                    Fermion,
                                    itops.vals2coefs(Gt.get_block(ind_path(1))),
                                    itops.vals2coefs(T),
                                    TIME_ORDERED);*/
                                T = convolve_rectangular(itops, 
                                    beta, 
                                    Gt.get_block(ind_path(1)),
                                    T);
                            }

                            // 3. for each kappa, multiply by F_kappa from right
                            auto Tkap = nda::zeros<dcomplex>(
                                num_Fs,
                                r,
                                F2_block.shape(0),
                                F1list[0].get_block_size(i, 1));
                            for (int kap = 0; kap < num_Fs; kap++) {
                                auto F1_block = F1list[kap].get_block(i);
                                for (int t = 0; t < r; t++) {
                                    Tkap(kap,t,_,_) = nda::matmul(T(t,_,_),F1_block);
                                }
                            }

                            // 4. for each mu, kap, mult by Delta_mu_kap and sum kap
                            auto Tmu = nda::make_regular(0*Tkap); 
                            // initialize Tmu with same shape as Tkap
                            for (int mu = 0; mu < num_Fs; mu++) {
                                for (int kap = 0; kap < num_Fs; kap++) {
                                    for (int t = 0; t < r; t++) {
                                        if (fb1) {
                                            Tmu(mu,t,_,_) += hyb(t,mu,kap)*Tkap(kap,t,_,_);
                                        } else {
                                            Tmu(mu,t,_,_) += hyb_refl(t,mu,kap)*Tkap(kap,t,_,_);
                                        }
                                    }
                                }
                            }

                            // 5. multiply by F^dag_mu and sum over mu
                            auto U = nda::make_regular(0*Tmu(0,_,_,_));
                            for (int mu = 0; mu < num_Fs; mu++) {
                                for (int t = 0; t < r; t++) {
                                    auto F3_block = F3list[mu].get_block(ind_path(1));
                                    U(t,_,_) += nda::matmul(F3_block, Tmu(mu,t,_,_));
                                }
                            }

                            auto Fbar_block = Fbar_array[lam][l].get_block(ind_path(2));
                            if (omega_l_is_pos) {
                                // 6. convolve by G K^+
                                auto GKplus = nda::make_regular(Gt.get_block(ind_path(2)));
                                for (int t = 0; t < r; t++) {
                                    GKplus(t,_,_) = Kplus(t,l)*Gt.get_block(ind_path(2))(t,_,_);
                                }
                                /*
                                U = itops.convolve(
                                    beta,
                                    Fermion,
                                    itops.vals2coefs(GKplus),
                                    itops.vals2coefs(U),
                                    TIME_ORDERED);*/
                                U = convolve_rectangular(itops, beta, GKplus, U);
                                // 7. multiply by Fbar                                
                                for (int t = 0; t < r; t++) {
                                    Sigma_l(t,_,_) += nda::matmul(Fbar_block, U(t,_,_));
                                }
                            }
                            else {
                                // 6. convolve by G
                                /*
                                U = itops.convolve(
                                    beta,
                                    Fermion,
                                    itops.vals2coefs(Gt.get_block(ind_path(2))),
                                    itops.vals2coefs(U),
                                    TIME_ORDERED);*/
                                U = convolve_rectangular(itops, beta, Gt.get_block(ind_path(2)), U);
                                // 7. multiply by K^+(tau) Fbar
                                for (int t = 0; t < r; t++) {
                                    Sigma_l(t,_,_) += Kplus(t,l)*nda::matmul(Fbar_block, U(t,_,_));
                                }
                            }
                        }
                        if (omega_l_is_pos) {
                            // Sigma.add_block(i, nda::make_regular(sfM*Sigma_l/k_it(0, dlr_rf(l))));
                            Sigma_block_i += nda::make_regular(sfM*Sigma_l/k_it(0, dlr_rf(l)));
                        }
                        else {
                            // Sigma.add_block(i, nda::make_regular(sfM*Sigma_l/k_it(0, -dlr_it(l))));
                            Sigma_block_i += nda::make_regular(sfM*Sigma_l/k_it(0, -dlr_rf(l)));
                        }
                    }
                }
                Sigma.add_block(i, Sigma_block_i);
            }
        }
    }

    return Sigma;
}

BlockDiagOpFun eval_backbone(double beta, 
    imtime_ops &itops, 
    const BlockOp &vertex_0, 
    const BlockOpFun &vertex_1, 
    const std::vector<BlockOp> &right_vertices, 
    const BlockOpFun &special_vertex, 
    const std::vector<BlockOp> &left_vertices, 
    const std::vector<BlockDiagOpFun> &edges, 
    nda::array_const_view<dcomplex,1> prefactor) {

    // vertex and edge numbering example:
    //
    // vertices: 5         4         3         2         1         0
    //                                                vertex_1  vertex_0
    //  right_v:                               0
    //                           special_v
    //   left_v: 1         0
    //           |         |         |         |         |         |
    //           ---------------------------------------------------
    //    (time: tau     tau_4     tau_3     tau_2     tau_1       0)
    //    edges:      3         2         1         0 
    // ind_path: 5    4    4    3    3    2    2    1    1         0 TODO: double check this!

    // TODO: exception if #vertices - #edges != 1
    int num_right_vertices = right_vertices.size();
    int num_left_vertices = left_vertices.size();
    int num_vertices = num_right_vertices + num_left_vertices + 1;
    int num_edges = edges.size();

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.shape(0);
    
    // initialize final result, this backbone's contribution to the self-energy
    int num_block_cols = edges[0].get_num_block_cols();
    auto Sigma_backbone = BlockDiagOpFun(r, edges[0].get_block_sizes());

    for (int i = 0; i < num_block_cols; i++) {

        // "backwards pass"
        // 
        // for each self-energy block, find contributing blocks of factors
        // 
        // paths_all_nonzero: false if for i-th block of 
        // Sigma, a block of one of the factors is zero
        // 
        // ind_path: a vector of column indices of the 
        // blocks of the factors that contribute. if paths_all_nonzero is 
        // false at this index, values in ind_path are garbage.
        //
        // ATTN: assumes all BlockOps in F(1,2,3)list have the same structure
        // i.e, index path is independent of kappa, mu
        bool path_all_nonzero = true;
        auto ind_path = nda::zeros<int>(num_vertices);
        auto Sigma_block_i = nda::make_regular(0*Sigma_backbone.get_block(i));

        int ip = vertex_0.get_block_index(i); 
        if (ip == -1) {path_all_nonzero = false;}
        else {
            ind_path(0) = ip; // block ip of vertex 0 contributes to block i of Sigma
            ip = vertex_1.get_block_index(ip);
        }
        // TODO: double-check, probably wrong
        int v = 2;
        while (v < num_vertices && path_all_nonzero) { // loop through vertices
            if (ip == -1 || edges[v-2].get_zero_block_index(ip) == -1) {
                path_all_nonzero = false; // next edge or vertex has zero block
            }
            else {
                if (v < num_right_vertices-2) {
                    ip = right_vertices[v].get_block_index(ip);
                } else if (v == num_right_vertices - 2) {
                    ip = special_vertex.get_block_index(ip);
                } else {
                    ip = left_vertices[v].get_block_index(ip);
                }
                ind_path(v) = ip;
            }
            v += 1;
        }

        if (path_all_nonzero) {
            // allocate intermediate matrices
            std::vector<nda::array<dcomplex,3>> int_mat(num_vertices-1);

            // 1. starting from tau_1, proceed right-to-left, performing 
            //    multiplications at vertices and convolutions at edges, until 
            //    reaching the vertex containing the undecomposed hybridization 
            //    line (special_vertex)
            // 
            //    convolve multiply convolve ... convolve
            
            auto vertex_1_block = vertex_1.get_block(ind_path(0));
            int_mat[0] = convolve_rectangular(
                itops, 
                beta, 
                edges[0].get_block(ind_path(0)), 
                vertex_1_block);

            /* 
            for (int v = 2; v < num_right_vertices+2; v++) {
                auto block_size_v = right_vertices[v].get_block_size(i);
                int_mat[1] = nda::zeros<dcomplex>(r, block_size_v(0), block_size_v(1));
                for (int t = 0; t < r; t++) {
                    T(t,_,_) = nda::matmul(right_vertices[v].get_block(ind_path(v)), 
                        edges[v-1].get_block(ind_path(v))(t,_,_));
                }
            }*/
        }
    }
    
    return Sigma_backbone;
}