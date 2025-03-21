#include <cppdlr/dlr_build.hpp>
#include <nda/declarations.hpp>
#include <nda/nda.hpp>
#include <block_sparse.hpp>
#include <cppdlr/cppdlr.hpp>

using namespace nda;

nda::matrix<dcomplex> hyb_fun(double beta, double t, double s) {
    auto hyb = nda::matrix<dcomplex>(2,2);

    hyb(0,0) = k_it(t, 2.3, beta) + k_it(t, -2.3, beta);
    hyb(0,1) = s*(k_it(t, 2.3, beta) + k_it(t, -2.3, beta));
    hyb(1,0) = hyb(0,1);
    hyb(1,1) = hyb(0,0);

    return hyb;
}

int main() {
    // set up arguments to block_sparse/NCA_bs()
    int N = 4;
    int n = 2;

    // set up hybridization function, see two-band Anderson impurity model
    double beta = 2; // inverse temperature
    double Lambda = 12.5*beta; // DLR cutoff
    double eps = 1e-8; // DLR error tolerance
    int ntst_t = 1000; // test pts in imaginary time
    int nmax_om = 1000; // test pts in imaginary frequency

    // get DLR frequencies
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    int r = dlr_rf.size(); // number of imaginary time nodes
    
    auto itops = imtime_ops(Lambda, dlr_rf); // DLR imaginary time object
    auto dlr_it = itops.get_itnodes(); // r imaginary time nodes
    auto hyb = nda::array<dcomplex,3>(r, n, n); // Green's f'n as ndarray
    double s = .5; // strength of off-diag hybridization
    for (int i = 0; i < r; i++) { hyb(i, _, _) = hyb_fun(beta, dlr_it(i), s); }

    auto hyb_coeff = itops.vals2coefs(hyb); // DLR coefficients of Green's f'n
    std::cout << hyb_coeff(r-1,_,_) << std::endl;

    // set up Green's function
    dcomplex mu = 0.2789;
    dcomplex U = 1.01;
    dcomplex V = 0.123;
    nda::array<dcomplex,3> block0({r, 1, 1});
    nda::array<dcomplex,3> block1({r, 2, 2});
    nda::array<dcomplex,3> block2({r, 1, 1});
    nda::matrix<double> oneroot2{{-1/sqrt(2), 1/sqrt(2)}, 
        {1/sqrt(2), 1/sqrt(2)}};
    for (int t = 0; t < r; ++t) {
        block0(t,0,0) = 1;

        block1(t,0,0) = exp(-t*(mu-V));
        block1(t,0,1) = 0;
        block1(t,1,0) = 0;
        block1(t,1,1) = exp(-t*(mu+V));
        block1(t,_,_) = nda::matmul(oneroot2, block1(t,_,_));
        block1(t,_,_) = nda::matmul(block1(t,_,_), oneroot2);

        block2(t,0,0) = exp(-t*(2*mu+U));
    }
    std::vector<nda::array<dcomplex,3>> Gt_blocks = 
        {block0, block1, block2};
    BlockDiagonalOperator Gt(Gt_blocks);

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

    std::vector<FOperator> Fs = {F_up, F_down};
    // BlockDiagonalOperator NCA_result = NCA_bs(hyb, Gt, Fs);

    // std::cout << "NCA_result = " << NCA_result << std::endl;

    // compute NCA_result using dense storage

    nda::array<dcomplex,3> Gt_dense({r, N, N});
    Gt_dense(_,0,0) = block0(_,0,0);
    Gt_dense(_,range(1,3),range(1,3)) = block1;
    Gt_dense(_,3,3) = block2(_,0,0);

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
    
    std::cout << "G(0) = " << Gt_dense(0,_,_) << std::endl;
    std::cout << "NCA_result_dense = " << NCA_result_dense(0,_,_) << std::endl;
    return 0;
};