#include <nda/declarations.hpp>
#include <nda/nda.hpp>
#include <strong_cpl.hpp>
#include <block_sparse.hpp>

using namespace nda;

int main() {
    // set up arguments to block_sparse/NCA_bs()
    int N = 4;
    int r = 1;
    int n = 2;
    // TODO: input hybridization into NCA_bs
    hyb_F Delta(N, r, n);
    hyb_F Delta_reflect(N, r, n);

    // set up Green's function
    dcomplex mu = 0.25;
    dcomplex U = 1.0;
    dcomplex V = 0.1;
    nda::array<dcomplex,3> block0({r, 1, 1});
    nda::array<dcomplex,3> block1({r, 2, 2});
    nda::array<dcomplex,3> block2({r, 1, 1});
    for (int t = 0; t < r; ++t) {
        block0(t,0,0) = 0;
        block1(t,0,0) = mu;
        block1(t,1,1) = mu;
        block1(t,0,1) = V;
        block1(t,1,0) = V;
        block2(t,0,0) = 2*mu+V;
    }
    std::vector<nda::array<dcomplex,3>> Gt_blocks = 
        {block0, block1, block2};
    DiagonalOperator Gt(Gt_blocks);

    // set up annihilation operators
    nda::vector<int> block_indices_F = {1, 2, -1};

    nda::array<dcomplex,2> F_up_block0 = {{1, 0}};
    nda::array<dcomplex,2> F_up_block1 = {{0}, {1}};
    nda::array<dcomplex,2> F_up_block2 = {{0}};
    std::vector<nda::array<dcomplex,2>> F_up_blocks = 
        {F_up_block0, F_up_block1, F_up_block2};
    FOperator F_up(block_indices_F, F_up_blocks);

    nda::array<dcomplex,2> F_down_block0 = {{0, 1}};
    nda::array<dcomplex,2> F_down_block1 = {{-1}, {0}};
    nda::array<dcomplex,2> F_down_block2 = {{0}};
    std::vector<nda::array<dcomplex,2>> F_down_blocks = 
        {F_down_block0, F_down_block1, F_down_block2};
    FOperator F_down(block_indices_F, F_down_blocks);

    DiagonalOperator NCA_result = NCA_bs(Gt, {F_up, F_down});

    std::cout << "NCA_result = " << NCA_result << std::endl;

    return 0;
};