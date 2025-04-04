#include <nda/declarations.hpp>
#include <nda/nda.hpp>
#include <block_sparse.hpp>

using namespace nda;

int main() {
    // set up arguments to block_sparse/NCA_bs()
    int N = 4;
    int r = 1;
    int n = 2;

    // set up hybridization
    nda::array<dcomplex,3> hyb({r, n, n});
    for (int t = 0; t < r; ++t) {
        hyb(t,0,0) = 1;
        hyb(t,1,1) = 1;
        hyb(t,0,1) = 1;
        hyb(t,1,0) = 1;
    }

    // set up Green's function
    dcomplex mu = 0.2789;
    dcomplex U = 1.01;
    dcomplex V = 0.123;
    nda::array<dcomplex,3> block0({r, 1, 1});
    nda::array<dcomplex,3> block1({r, 2, 2});
    nda::array<dcomplex,3> block2({r, 1, 1});
    for (int t = 0; t < r; ++t) {
        block0(t,0,0) = 0;
        block1(t,0,0) = mu;
        block1(t,1,1) = mu;
        block1(t,0,1) = V;
        block1(t,1,0) = V;
        block2(t,0,0) = 2*mu+U;
    }
    std::vector<nda::array<dcomplex,3>> Gt_blocks = 
        {block0, block1, block2};
    BlockDiagOpFun Gt(Gt_blocks);

    // set up annihilation operators
    nda::vector<int> block_indices_F = {1, 2, -1};

    nda::array<dcomplex,2> F_up_block0 = {{1, 0}};
    nda::array<dcomplex,2> F_up_block1 = {{0}, {1}};
    nda::array<dcomplex,2> F_up_block2 = {{0}};
    std::vector<nda::array<dcomplex,2>> F_up_blocks = 
        {F_up_block0, F_up_block1, F_up_block2};
    BlockOp F_up(block_indices_F, F_up_blocks);

    nda::array<dcomplex,2> F_down_block0 = {{0, 1}};
    nda::array<dcomplex,2> F_down_block1 = {{-1}, {0}};
    nda::array<dcomplex,2> F_down_block2 = {{0}};
    std::vector<nda::array<dcomplex,2>> F_down_blocks = 
        {F_down_block0, F_down_block1, F_down_block2};
    BlockOp F_down(block_indices_F, F_down_blocks);

    std::vector<BlockOp> Fs = {F_up, F_down};
    BlockDiagOpFun NCA_result = NCA_bs(hyb, Gt, Fs);

    std::cout << "NCA_result = " << NCA_result << std::endl;

    // compute NCA_result using dense storage

    nda::array<dcomplex,3> Gt_dense({r, N, N});
    Gt_dense(0,0,0) = 0;
    Gt_dense(0,1,1) = mu;
    Gt_dense(0,2,2) = mu;
    Gt_dense(0,1,2) = V;
    Gt_dense(0,2,1) = V;
    Gt_dense(0,3,3) = 2*mu+U;

    nda::array<dcomplex,2> F_up_dense({N, N});
    F_up_dense(0,1) = 1;
    F_up_dense(2,3) = 1;

    nda::array<dcomplex,2> F_down_dense({N, N});
    F_down_dense(0,2) = 1;
    F_down_dense(1,3) = -1;

    nda::array<dcomplex,2> F_up_dag_dense = nda::transpose(F_up_dense);
    nda::array<dcomplex,2> F_down_dag_dense = nda::transpose(F_down_dense);

    nda::array<dcomplex,3> NCA_result_dense({r, N, N});
    nda::array<dcomplex,2> temp_dense({N, N});
    NCA_result_dense = 0;
    for (int t = 0; t < r; ++t) {
        // forward diagram
        temp_dense = nda::matmul(F_up_dag_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) += nda::matmul(temp_dense, F_up_dense);
        temp_dense = nda::matmul(F_up_dag_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) += nda::matmul(temp_dense, F_down_dense);
        temp_dense = nda::matmul(F_down_dag_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) += nda::matmul(temp_dense, F_up_dense);
        temp_dense = nda::matmul(F_down_dag_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) += nda::matmul(temp_dense, F_down_dense);

        // backward diagram
        temp_dense = nda::matmul(F_up_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= nda::matmul(temp_dense, F_up_dag_dense);
        temp_dense = nda::matmul(F_up_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= nda::matmul(temp_dense, F_down_dag_dense);
        temp_dense = nda::matmul(F_down_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= nda::matmul(temp_dense, F_up_dag_dense);
        temp_dense = nda::matmul(F_down_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= nda::matmul(temp_dense, F_down_dag_dense);
    }

    // std::cout << "Gt_dense = " << Gt_dense(0,_,_) << std::endl;
    // std::cout << "F_up_dense = " << F_up_dense << std::endl;
    // std::cout << "F_down_dense = " << F_down_dense << std::endl;
    // std::cout << "F_up_dag_dense = " << F_up_dag_dense << std::endl;
    // std::cout << "F_down_dag_dense = " << F_down_dag_dense << std::endl;
    std::cout << "NCA_result_dense = " << NCA_result_dense(0,_,_) << std::endl;
    return 0;
};