#include "block_sparse.hpp"
#include <initializer_list>
#include <iostream>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/print.hpp>

using namespace nda;

DiagonalOperator::DiagonalOperator(
    std::vector<nda::array<dcomplex,3>> &blocks) : blocks(blocks) {}

FOperator::FOperator(
    nda::vector<int> &block_indices, 
    std::vector<nda::array<dcomplex,2>> &blocks) : 
        block_indices(block_indices), blocks(blocks) {}

BlockOperator::BlockOperator(
    nda::vector<int> &block_indices, 
    std::vector<nda::array<dcomplex,3>> &blocks) : 
        block_indices(block_indices), blocks(blocks) {}

std::ostream& operator<<(std::ostream& os, DiagonalOperator &D) {
    // Print DiagonalOperator
    // @param[in] os output stream
    // @param[in] D DiagonalOperator
    // @return output stream

    int num_blocks = D.num_blocks;
    for (int i = 0; i < num_blocks; ++i) {
        os << "Block " << i << ":\n" << D.blocks[i] << "\n";
    }
    return os;
};

std::ostream& operator<<(std::ostream& os, FOperator &F) {
    // Print FOperator
    // @param[in] os output stream
    // @param[in] F FOperator
    // @return output stream

    int num_blocks = F.num_blocks;
    os << "Block indices: " << F.block_indices << "\n";
    for (int i = 0; i < num_blocks; ++i) {
        os << "Block " << i << ":\n" << F.blocks[i] << "\n";
    }
    return os;
};

FOperator dagger_bs(FOperator &F) {
    // Evaluate F^dagger in block-sparse storage
    // @param[in] F F operator
    // @return F^dagger operator

    int num_blocks = F.num_blocks;
    int i, j;

    // find block indices for F^dagger
    nda::vector<int> block_indices_dag(num_blocks);
    // initialiaze indices with -1
    block_indices_dag = -1;
    std::vector<nda::array<dcomplex,2>> blocks_dag(num_blocks);
    for (i = 0; i < num_blocks; ++i) {
        j = F.block_indices[i];
        if (j > -1) {
            block_indices_dag[j] = i;
            blocks_dag[j] = nda::transpose(F.blocks[i]);
        }
    }
    FOperator F_dag(block_indices_dag, blocks_dag);
    return F_dag;
}

// BlockOperator NCA_bs(hyb_F &hyb_self, hyb_F &hyb_reflect, 
DiagonalOperator NCA_bs(
    DiagonalOperator &Gt, std::initializer_list<FOperator> F_list) {
    // Evaluate matrix products in NCA using block-sparse storage
    // @param[in] hyb_self hyb_F object for Delta(t)
    // @param[in] hyb_reflect hyb_F object for Delta(beta-t)
    // @param[in] Gt Greens function
    // @param[in] F_list F operators
    // @return NCA self-energy
    
    std::vector<FOperator> Fs(F_list);
    // get F^dagger operators
    int num_Fs = Fs.size();
    std::vector<FOperator> F_dags(F_list);
    for (int i = 0; i < num_Fs; ++i) {
        F_dags[i] = dagger_bs(Fs[i]);
    }

    // initialize blocks of self-energy, with same shape as Gt
    std::vector<nda::array<dcomplex,3>> diag_blocks(Gt.blocks);
    int num_blocks = Gt.num_blocks;
    for (int i = 0; i < num_blocks; ++i) {
        diag_blocks[i] = 0; 
    }
    int r = Gt.blocks[0].shape(0); // number of time indices
    for (int t = 0; t < r; ++t) {
        // forward diagram contribution to self-energy
        for (FOperator &F_dag : F_dags) {
            for (FOperator &F : Fs) {
                for (int i = 0; i < num_blocks; ++i) {
                    int j = F_dag.block_indices[i]; // = col ind of block i
                    if (j > -1) { // if F_dag has block in row i
                        auto temp = nda::matmul(
                            F_dag.blocks[i], Gt.blocks[j](t,_,_));
                        auto prod_block = nda::matmul(
                            temp, F.blocks[j]);
                        diag_blocks[i](t,_,_) += prod_block;
                    }
                }
            }
        }
        // backward diagram contribution to self-energy
        for (FOperator &F : Fs) {
            for (FOperator &F_dag : F_dags) {
                for (int i = 0; i < num_blocks; ++i) {
                    int j = F.block_indices[i]; // = col ind of block i
                    if (j > -1) { // if F has block in row i
                        auto temp = nda::matmul(
                            F.blocks[i], Gt.blocks[j](t,_,_));
                        auto prod_block = nda::matmul(
                            temp, F_dag.blocks[j]);
                        diag_blocks[i](t,_,_) -= prod_block;
                    }
                }
            }
        }
    }

    DiagonalOperator Sigma(diag_blocks);
    return Sigma;
}