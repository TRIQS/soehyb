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

BlockDiagonalOperator::BlockDiagonalOperator(
    std::vector<nda::array<dcomplex,3>> &blocks) : blocks(blocks) {
        
    num_block_rows = blocks.size(); 
    }

void BlockDiagonalOperator::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {
        
    this->blocks = blocks;
    num_block_rows = blocks.size();
    }

const std::vector<nda::array<dcomplex,3>>& BlockDiagonalOperator::get_blocks() const {        
    return blocks;
    }

const nda::array<dcomplex,3>& BlockDiagonalOperator::get_block(int i) const {
    return blocks[i];
    }

const int BlockDiagonalOperator::get_num_block_rows() const {
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

const nda::vector<int>& FOperator::get_block_indices() const {
    return block_indices;
    }

int FOperator::get_block_index(int i) const {
    return block_indices(i);
}

const std::vector<nda::array<dcomplex,2>>& FOperator::get_blocks() const {
    return blocks;
}

const nda::array<dcomplex,2>& FOperator::get_block(int i) const {
    if (block_indices[i] == -1) {
        static auto arr = nda::zeros<dcomplex>(1,1);
        return arr;
    }
    else {
        return blocks[i];
    }
}

const int FOperator::get_num_block_rows() const {
    return num_block_rows;
}

const int FOperator::get_num_blocks() const {
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

const nda::vector<int>& BlockOperator::get_block_indices() const {
    return block_indices;
}

int BlockOperator::get_block_index(int i) const {
    return block_indices(i);
}

const std::vector<nda::array<dcomplex,3>>& BlockOperator::get_blocks() const {
    return blocks;
}

const nda::array<dcomplex,3>& BlockOperator::get_block(int i) const {
    if (block_indices[i] == -1) {
        static auto arr = nda::zeros<dcomplex>(1,1,1);
        return arr;
    }
    else {
        return blocks[i];
    }
}

const int BlockOperator::get_num_block_rows() const {
    return num_block_rows;
}

const int BlockOperator::get_num_blocks() const {
    return num_blocks;
}


std::ostream& operator<<(std::ostream& os, BlockDiagonalOperator &D) {
    // Print BlockDiagonalOperator
    // @param[in] os output stream
    // @param[in] D BlockDiagonalOperator
    // @return output stream

    for (int i = 0; i < D.get_num_block_rows(); ++i) {
        os << "Block " << i << ":\n" << D.get_block(i) << "\n";
    }
    return os;
};

std::ostream& operator<<(std::ostream& os, FOperator &F) {
    // Print FOperator
    // @param[in] os output stream
    // @param[in] F FOperator
    // @return output stream

    os << "Block indices: " << F.get_block_indices() << "\n";
    for (int i = 0; i < F.get_num_block_rows(); ++i) {
        if (F.get_block_indices()[i] == -1) {
            os << "Block " << i << ": 0\n";
        }
        else {
            os << "Block " << i << ":\n" << F.get_block(i) << "\n";
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

BlockOperator operator*(
    const BlockDiagonalOperator& A, 
    const BlockOperator& B) {
    // Compute a product between a BlockDiagonalOperator and a BlockOperator
    // @param[in] A BlockDiagonalOperator
    // @param[in] B BlockOperator

    // initialize blocks of product, which has same shape as B
    auto B_blocks = B.get_blocks();
    std::vector<nda::array<dcomplex,3>> prod_blocks(B_blocks);

    auto B_block_indices = B.get_block_indices();
    BlockOperator product(B_block_indices, B_blocks);
    int r = B_blocks[0].shape(0);
    if (r != A.get_block(0).shape(0)) {
        throw std::invalid_argument("number of time indices do not match");
    }
    for (int i = 0; i < A.get_num_block_rows(); ++i) {
        if (B_block_indices(i) == -1) { // block-row i has no nonzero block
            prod_blocks[i] = 0;
        }
        else {
            for (int t = 0; t < r; ++t) {
                prod_blocks[i](t,_,_) = nda::matmul(
                    A.get_block(i)(t,_,_), B_blocks[i](t,_,_));
            }
        }
    }

    product.set_blocks(prod_blocks);
    return product;
}

BlockOperator operator*(
    const BlockOperator& A, 
    const BlockDiagonalOperator& B) {
    // Compute a product between a BlockOperator and a BlockDiagonalOperator
    // @param[in] A BlockOperator
    // @param[in] B BlockDiagonalOperator

    // initialize blocks of product, which has same shape as B
    auto A_blocks = A.get_blocks();
    std::vector<nda::array<dcomplex,3>> prod_blocks(A_blocks);

    auto A_block_indices = A.get_block_indices();
    BlockOperator product(A_block_indices, A_blocks);
    int r = A_blocks[0].shape(0);
    if (r != B.get_block(0).shape(0)) {
        throw std::invalid_argument("number of time indices do not match");
    }
    for (int i = 0; i < B.get_num_block_rows(); ++i) {
        int j = A_block_indices(i);
        if (j == -1) { // block-row i has no nonzero block
            prod_blocks[i] = 0;
        }
        else {
            for (int t = 0; t < r; ++t) {
                prod_blocks[i](t,_,_) = nda::matmul(
                    A_blocks[i](t,_,_), B.get_block(j)(t,_,_));
            }
        }
    }

    product.set_blocks(prod_blocks);
    return product;
}

BlockOperator operator*(const BlockDiagonalOperator& A, const FOperator& F) {
    // Compute a product between a BlockDiagonalOperator and an FOperator
    // @param[in] A BlockDiagonalOperator
    // @param[in] F FOperator

    auto F_blocks = F.get_blocks();
    std::vector<nda::array<dcomplex,3>> prod_blocks;
    int r = A.get_block(0).shape(0);
    auto F_block_indices = F.get_block_indices();

    // initialize shapes of arrays in prod_blocks
    for (int i = 0; i < A.get_num_block_rows(); ++i) {
        if (F_block_indices(i) == -1) {
            prod_blocks.emplace_back(nda::zeros<dcomplex>(r,1,1));
        }
        else {
            auto F_shape = F_blocks[i].shape();
            auto prod_block = nda::zeros<dcomplex>(r,F_shape[0],F_shape[1]);
            prod_blocks.emplace_back(prod_block);
        }
    }
    
    for (int i = 0; i < A.get_num_block_rows(); ++i) {
        if (F_block_indices(i) != -1) {
            for (int t = 0; t < r; ++t) {
                prod_blocks[i](t,_,_) = nda::matmul(
                    A.get_block(i)(t,_,_), F_blocks[i]);
            }
        }
    }

    BlockOperator product(F_block_indices, prod_blocks);
    return product;
}

BlockOperator operator*(const FOperator& F, const BlockDiagonalOperator& B) {
    // Compute a product between an FOperator and a BlockDiagonalOperator
    // @param[in] F FOperator
    // @param[in] B BlockDiagonalOperator

    auto F_blocks = F.get_blocks();
    std::vector<nda::array<dcomplex,3>> prod_blocks;
    int r = B.get_block(0).shape(0);
    auto F_block_indices = F.get_block_indices();

    std::cout << "here 5" << std::endl;

    // initialize shapes of arrays in prod_blocks
    for (int i = 0; i < B.get_num_block_rows(); ++i) {
        if (F_block_indices(i) == -1) {
            prod_blocks.emplace_back(nda::zeros<dcomplex>(r,1,1));
        }
        else {
            auto F_shape = F_blocks[i].shape();
            auto prod_block = nda::zeros<dcomplex>(r,F_shape[0],F_shape[1]);
            prod_blocks.emplace_back(prod_block);
        }
    }
    
    for (int i = 0; i < B.get_num_block_rows(); ++i) {
        int j = F_block_indices(i);
        if (j != -1) {
            for (int t = 0; t < r; ++t) {
                std::cout << F_blocks[i] << std::endl;
                std::cout << B.get_block(j)(t,_,_) << std::endl;
                prod_blocks[i](t,_,_) = nda::matmul(
                    F_blocks[i], B.get_block(j)(t,_,_));
            }
        }
    }

    BlockOperator product(F_block_indices, prod_blocks);
    return product;
}

BlockOperator operator*(
    nda::vector_const_view<dcomplex>& f, 
    const BlockOperator& A) {
    // Compute a product between a scalar f'n of time and a BlockOperator
    // @param[in] f nda::array_const_view<dcomplex,1>
    // @param[in] A BlockOperator
    // @return product

    int num_block_rows = A.get_num_block_rows();
    auto prod_blocks(A.get_blocks());
    int r = prod_blocks[0].shape(0);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < num_block_rows; j++) {
            prod_blocks[j](i,_,_) = f(i)*prod_blocks[j](i,_,_); 
        }
    }
    auto block_indices = A.get_block_indices();
    BlockOperator product(block_indices,prod_blocks);
    return product;
}

BlockOperator operator*(
    const BlockOperator& A,
    nda::vector_const_view<dcomplex>& f) {
    // Compute a product between a scalar f'n of time and a BlockOperator
    // @param[in] A BlockOperator
    // @param[in] f nda::array_const_view<dcomplex,1>
    // @return product

    int num_block_rows = A.get_num_block_rows();
    auto prod_blocks(A.get_blocks());
    int r = prod_blocks[0].shape(0);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < num_block_rows; j++) {
            prod_blocks[j](i,_,_) = f(i)*prod_blocks[j](i,_,_); 
        }
    }
    auto block_indices = A.get_block_indices();
    BlockOperator product(block_indices,prod_blocks);
    return product;
}

BlockDiagonalOperator NCA_bs(
    nda::array_const_view<dcomplex,3> hyb, 
    BlockDiagonalOperator const &Gt, 
    const std::vector<FOperator> &Fs) {
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
    
    BlockDiagonalOperator Sigma(diag_blocks);
    int r = Gt.get_blocks()[0].shape(0); // number of time indices
    for (int t = 0; t < r; ++t) {
        // forward diagram contribution to self-energy
        // make loop over forward/backward for higher order diagrams 
        for (int l = 0; l < num_Fs; ++l) {
            FOperator const &F_dag = F_dags[l]; 
            for (int k = 0; k < num_Fs; ++k) {
                FOperator const &F = Fs[k];
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
            FOperator const &F = Fs[l];
            for (int k = 0; k < num_Fs; ++k) {
                FOperator const &F_dag = F_dags[k];
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

BlockDiagonalOperator OCA_bs(
    nda::array_const_view<dcomplex, 3> hyb_coeffs, 
    nda::vector_const_view<double> dlr_rf, 
    nda::vector_const_view<double> dlr_it, 
    imtime_ops &itops, 
    double beta, 
    const BlockDiagonalOperator &Gt, 
    const std::vector<FOperator> &Fs) {
    // Evaluate OCA using block-sparse storage
    // @param[in] hyb_coeffs DLR coefficients of hybridization
    // @param[in] dlr_it DLR imaginary time nodes
    // @param[in] dlr_rf DLR real frequencies
    // @param[in] itops cppdlr imaginary time object
    // @param[in] beta inverse temperature
    // @param[in] Gt Greens function at times dlr_it
    // @param[in] F F operator
    // @param[in] F_dag F^dagger operator
    // @return OCA term of self-energy

    // number of imaginary time nodes
    int r = dlr_it.shape(0);

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

    // get DLR coefficients of Gt
    // auto Gtc = itops.vals2coefs(Gt); // TODO: compute DLR coeffs of BlockDiagonalOperator

    // evaluate matrix K with (k,l)-entry K(tau_k,omega_l)
    auto K = K_mat(dlr_it, dlr_rf);

    BlockDiagonalOperator Sigma(diag_blocks);

    // loop over hybridization lines
    for (int fb1 = 1; fb1 > -1; fb1--) {
        for (int fb2 = 1; fb2 > -1; fb2--) {
            // fb = 1 for forward line, else = 0
            // fb1 corresponds with line from 0 to tau_2
            std::vector<FOperator> const &F1list = (fb1) ? Fs : F_dags;
            std::vector<FOperator> const &F2list = (fb2) ? Fs : F_dags;
            std::vector<FOperator> const &F3list = (fb1) ? F_dags : Fs;
            std::vector<FOperator> const &F4list = (fb2) ? F_dags : Fs;
            int sfM = (fb1^fb2) ? 1 : -1; // sign

            // omega_l <= sum
            for (int l = 0; l < r; ++l) {
                for (int lam = 0; lam < num_Fs; ++lam) {
                    FOperator const &F2 = F2list[lam]; 
                    // 1. multiply F_lambda G(tau_1) K^-(tau_1)
                    BlockOperator integrand = F2*Gt;
                    vector_const_view<double> Kl = K(_,l);
                    integrand = Kl*integrand; 
                    // 2. convolve by G
                    // TODO
                    
                    // 3. for each kappa, multiply my F_kappa from right
                    for (int kap = 0; kap < num_Fs; ++kap) {
                        FOperator const &F1 = F1list[kap];
                        // integrand = integrand*F1; // TODO: define BlockOp times FOp
                    }
                }
            }
        }
    }

    /* for (int t = 0; t < r; ++t) {
        // loop over placements of Fs in forward/backward hybr. propagations
        for (int nperm = 0; nperm < 4; ++nperm) { 
            // nperm = 0 -> Fdag Fdag F F
            // nperm = 1 -> Fdag F F Fdag
            // nperm = 2 -> F Fdag ...
            // nperm = 3 -> F F ...
            std::vector<FOperator> const &F1list = (nperm & 2) ? Fs : F_dags;
            std::vector<FOperator> const &F2list = (nperm & 1) ? Fs : F_dags;
            int sfM = (nperm == 0 || nperm == 3) ? -1 : 1; // sign

            for (int l = 0; l < num_Fs; ++l) {
                FOperator const &F1 = F1list[l]; 
                for (int k = 0; k < num_Fs; ++k) {
                    FOperator const &F2 = F2list[k];
                    for (int i = 0; i < num_block_rows; ++i) {
                        int j = F1.get_block_indices()[i]; // = col ind of block i
                        if (j != -1) { // if F_dag has block in row i
                            auto temp = nda::matmul(
                                F1.get_blocks()[i], Gt.get_blocks()[j](t,_,_));
                            auto prod_block = nda::matmul(
                                temp, F2.get_blocks()[j]);
                            diag_blocks[i](t,_,_) += sfM*hyb(t,l,k)*prod_block;
                        }
                    }
                }
            }
        }
    } */

    // compute contribution of this forward/backward combination
    // following "Automated evaluation...", sec IV.A.

    Sigma.set_blocks(diag_blocks);
    return Sigma;
}