#include "block_sparse.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <iostream>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/print.hpp>

using namespace nda;

DiagonalOperator::DiagonalOperator(
    std::vector<nda::array<dcomplex,3>> &blocks) : blocks(blocks) {
        num_block_rows = blocks.size(); 
    }

void DiagonalOperator::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {
        this->blocks = blocks;
        num_block_rows = blocks.size();
    }

std::vector<nda::array<dcomplex,3>> DiagonalOperator::get_blocks() const {
        return blocks;
    }

int DiagonalOperator::get_num_block_rows() const {
        return num_block_rows;
    }


FOperator::FOperator(
    nda::vector<int> &block_indices, 
    std::vector<nda::array<dcomplex,2>> &blocks) : 
    block_indices(block_indices), blocks(blocks) {
        num_block_rows = block_indices.size();
        num_blocks = std::count_if(
            block_indices.begin(), block_indices.end(), 
            [](int i) { return i != -1; });
    }

void FOperator::set_block_indices(
    nda::vector<int> &block_indices) {
        this->block_indices = block_indices;
        num_block_rows = block_indices.size();
        num_blocks = std::count_if(
            block_indices.begin(), block_indices.end(), 
            [](int i) { return i != -1; });
    }

void FOperator::set_blocks(
    std::vector<nda::array<dcomplex,2>> &blocks) {
        this->blocks = blocks;
        num_block_rows = blocks.size();
    }

nda::vector<int> FOperator::get_block_indices() const {
        return block_indices;
    }

std::vector<nda::array<dcomplex,2>> FOperator::get_blocks() const {
        return blocks;
    }

int FOperator::get_num_block_rows() const {
        return num_block_rows;
    }

int FOperator::get_num_blocks() const {
        return num_blocks;
    }


BlockOperator::BlockOperator(
    nda::vector<int> &block_indices, 
    std::vector<nda::array<dcomplex,3>> &blocks) : 
    block_indices(block_indices), blocks(blocks) {
        num_block_rows = block_indices.size();
        num_blocks = std::count_if(
            block_indices.begin(), block_indices.end(), 
            [](int i) { return i != -1; });
    }

void BlockOperator::set_block_indices(
    nda::vector<int> &block_indices) {
        this->block_indices = block_indices;
        num_block_rows = block_indices.size();
        num_blocks = std::count_if(
            block_indices.begin(), block_indices.end(), 
            [](int i) { return i != -1; });
    }

void BlockOperator::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {
        this->blocks = blocks;
        num_block_rows = blocks.size();
    }

nda::vector<int> BlockOperator::get_block_indices() const {
        return block_indices;
    }

std::vector<nda::array<dcomplex,3>> BlockOperator::get_blocks() const {
        return blocks;
    }

int BlockOperator::get_num_block_rows() const {
        return num_block_rows;
    }

int BlockOperator::get_num_blocks() const {
        return num_blocks;
    }


std::ostream& operator<<(std::ostream& os, DiagonalOperator &D) {
    // Print DiagonalOperator
    // @param[in] os output stream
    // @param[in] D DiagonalOperator
    // @return output stream

    int num_block_rows = D.get_num_block_rows();
    std::vector<nda::array<dcomplex,3>> blocks = D.get_blocks();
    for (int i = 0; i < num_block_rows; ++i) {
        os << "Block " << i << ":\n" << blocks[i] << "\n";
    }
    return os;
};

std::ostream& operator<<(std::ostream& os, FOperator &F) {
    // Print FOperator
    // @param[in] os output stream
    // @param[in] F FOperator
    // @return output stream

    int num_block_rows = F.get_num_block_rows();
    os << "Block indices: " << F.get_block_indices() << "\n";
    for (int i = 0; i < num_block_rows; ++i) {
        if (F.get_block_indices()[i] == -1) {
            os << "Block " << i << ": 0\n";
        }
        else {
            os << "Block " << i << ":\n" << F.get_blocks()[i] << "\n";
        }
    }
    return os;
};

FOperator dagger_bs(FOperator const &F) {
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
    FOperator F_dag(block_indices_dag, blocks_dag);
    return F_dag;
}

// notes
// =====
// initializer lists --> &vectors
// const &

DiagonalOperator NCA_bs(
    nda::array_const_view<dcomplex,3> hyb, 
    DiagonalOperator const &Gt, 
    const std::vector<FOperator> &Fs) {
    // Evaluate matrix products in NCA using block-sparse storage
    // @param[in] hyb_self hybridization function
    // @param[in] Gt Greens function
    // @param[in] F_list F operators
    // @return NCA self-energy
    
    // std::vector<FOperator> F_dags(Fs);
    // get F^dagger operators
    int num_Fs = Fs.size();
    
    // std::vector<FOperator> F_dags(F_list);
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
    
    DiagonalOperator Sigma(diag_blocks);
    int r = Gt.get_blocks()[0].shape(0); // number of time indices
    for (int t = 0; t < r; ++t) {
        // forward diagram contribution to self-energy
        // make loop over forward/backward for higher order diagrams 
        for (int l = 0; l < num_Fs; ++l) {
            FOperator const &F_dag = F_dags[l]; 
            for (int k = 0; k < num_Fs; ++k) {
                FOperator const &F = Fs[k];
                for (int i = 0; i < num_block_rows; ++i) {
                    int j = F_dag.get_block_indices()[i]; // = col ind of block i
                    if (j != -1) { // if F_dag has block in row i
                        auto temp = nda::matmul(
                            F_dag.get_blocks()[i], Gt.get_blocks()[j](t,_,_));
                        auto prod_block = nda::matmul(
                            temp, F.get_blocks()[j]);
                        diag_blocks[i](t,_,_) += nda::make_regular(
                            hyb(t,l,k)*prod_block);
                    }
                }
            }
        }
        // backward diagram contribution to self-energy
        for (int l = 0; l < num_Fs; ++l) {
            FOperator const &F = Fs[l];
            for (int k = 0; k < num_Fs; ++k) {
                FOperator const &F_dag = F_dags[k];
                for (int i = 0; i < num_block_rows; ++i) {
                    int j = F.get_block_indices()[i]; // = col ind of block i
                    if (j != -1) { // if F has block in row i
                        auto temp = nda::matmul(
                            F.get_blocks()[i], Gt.get_blocks()[j](t,_,_));
                        auto prod_block = nda::matmul(
                            temp, F_dag.get_blocks()[j]);
                        diag_blocks[i](t,_,_) -= nda::make_regular(
                            hyb(t,l,k)*prod_block);
                    }
                }
            }
        }
    }
    
    Sigma.set_blocks(diag_blocks);
    return Sigma;
}