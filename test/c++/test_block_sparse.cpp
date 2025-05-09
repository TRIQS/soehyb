#include <nda/declarations.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/nda.hpp>
#include <block_sparse.hpp>
#include <gtest/gtest.h>

using namespace nda;

TEST(BlockSparseNCATest, NCA) {
    // set up arguments to block_sparse/NCA_bs()
    int N = 4;
    int r = 1;
    int n = 2;

    // set up hybridization
    nda::array<dcomplex,3> hyb({r, n, n});
    for (int t = 0; t < r; ++t) {
        hyb(t,0,0) = 1;
        hyb(t,1,1) = -1;
        hyb(t,0,1) = -1;
        hyb(t,1,0) = 4;
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
    nda::vector<int> zero_block_indices = {-1, 0, 0};
    BlockDiagOpFun Gt(Gt_blocks, zero_block_indices);

    // set up annihilation operators
    nda::vector<int> block_indices_F = {-1, 0, 1};

    nda::array<dcomplex,2> F_up_block0 = {{0}};
    nda::array<dcomplex,2> F_up_block1 = {{1, 0}};
    nda::array<dcomplex,2> F_up_block2 = {{0}, {1}};
    std::vector<nda::array<dcomplex,2>> F_up_blocks = 
        {F_up_block0, F_up_block1, F_up_block2};
    BlockOp F_up(block_indices_F, F_up_blocks);

    nda::array<dcomplex,2> F_down_block0 = {{0}};
    nda::array<dcomplex,2> F_down_block1 = {{0, 1}};
    nda::array<dcomplex,2> F_down_block2 = {{-1}, {0}};
    std::vector<nda::array<dcomplex,2>> F_down_blocks = 
        {F_down_block0, F_down_block1, F_down_block2};
    BlockOp F_down(block_indices_F, F_down_blocks);

    std::vector<BlockOp> Fs = {F_up, F_down};
    BlockDiagOpFun NCA_result = NCA_bs(hyb, hyb, Gt, Fs);

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

    auto NCA_result_dense = nda::zeros<dcomplex>(r, N, N);
    nda::array<dcomplex,2> temp_dense({N, N});
    for (int t = 0; t < r; ++t) {
        // backward diagram
        temp_dense = nda::matmul(F_up_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) +=
            hyb(0,0,0)*nda::matmul(temp_dense, F_up_dag_dense);
        temp_dense = nda::matmul(F_up_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) += 
            hyb(0,0,1)*nda::matmul(temp_dense, F_down_dag_dense);
        temp_dense = nda::matmul(F_down_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) += 
            hyb(0,1,0)*nda::matmul(temp_dense, F_up_dag_dense);
        temp_dense = nda::matmul(F_down_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) += 
            hyb(0,1,1)*nda::matmul(temp_dense, F_down_dag_dense);

        // forward diagram
        temp_dense = nda::matmul(F_up_dag_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= 
            hyb(0,0,0)*nda::matmul(temp_dense, F_up_dense);
        temp_dense = nda::matmul(F_up_dag_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= 
            hyb(0,0,1)*nda::matmul(temp_dense, F_down_dense);
        temp_dense = nda::matmul(F_down_dag_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= 
            hyb(0,1,0)*nda::matmul(temp_dense, F_up_dense);
        temp_dense = nda::matmul(F_down_dag_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) = 
            hyb(0,1,1)*nda::matmul(temp_dense, F_down_dense);
    }

    EXPECT_EQ(NCA_result.get_blocks()[0](_,0,0), NCA_result_dense(_,0,0));
    EXPECT_EQ(NCA_result.get_blocks()[1], NCA_result_dense(_,range(1,3),range(1,3)));
    EXPECT_EQ(NCA_result.get_blocks()[2](_,0,0), NCA_result_dense(_,3,3));
}

// TODO: another NCA test
TEST(BlockSparseNCATest, NCA_single_exponential) {
    // DLR parameters
    double beta = 1.0;
    double Lambda = 100.0;
    double eps = 1.0e-13;
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);
    auto const & dlr_it = itops.get_itnodes();
    int r = itops.rank();

    auto dlr_it_abs = cppdlr::rel2abs(dlr_it);

    // create hybridization
    double D = 1.0;
    auto Deltat = nda::array<dcomplex,3>(r,1,1);
    auto Deltat_refl = nda::array<dcomplex,3>(r,1,1);
    Deltat(_,0,0) = exp(-D*dlr_it_abs*beta);
    Deltat_refl(_,0,0) = exp(D*dlr_it_abs*beta);

    // create Green's function
    double g = -1.0;
    auto Gt_block = nda::array<dcomplex,3>(r,1,1);
    auto Gt_zero_block_index = nda::ones<int>(1);
    Gt_block(_,0,0) = exp(-g*dlr_it_abs*beta);
    std::vector<nda::array<dcomplex,3>> Gt_blocks = {Gt_block};
    auto Gt = BlockDiagOpFun(Gt_blocks, Gt_zero_block_index);

    // create annihilation operator
    auto F_block = nda::ones<dcomplex>(1,1);
    auto F_block_indices = nda::vector<int>(1);
    F_block_indices = 0;
    std::vector<nda::array<dcomplex,2>> F_blocks = {F_block};
    auto F = BlockOp(F_block_indices, F_blocks);
    std::vector<BlockOp> Fs = {F};

    BlockDiagOpFun NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);
    auto NCA_ana = nda::zeros<dcomplex>(r);
    for (int i = 0; i < r; i++) {
        auto tau = dlr_it_abs(i);
        NCA_ana(i) = exp(2*tau) - 1;
    }

    EXPECT_LT(nda::norm(NCA_result.get_block(0)(_,0,0)-NCA_ana), 1.0e-10);
}

TEST(BlockSparseOCATest, OCA_matmuls) {
    // set up arguments to block_sparse/OCA_bs()
    double beta = 100;
    double eps = 1e-10;
    double Lambda = 100;
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);

    int N = 4;
    int r = dlr_rf.size();
    int n = 2;

    // set up hybridization
    nda::array<dcomplex,3> hyb({r, n, n});
    for (int t = 0; t < r; ++t) {
        hyb(t,0,0) = 1;
        hyb(t,1,1) = -1;
        hyb(t,0,1) = -1;
        hyb(t,1,0) = 4;
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
    nda::vector<int> zero_block_indices = {-1, 0, 0};
    BlockDiagOpFun Gt(Gt_blocks, zero_block_indices);

    // set up annihilation operators
    nda::vector<int> block_indices_F = {-1, 0, 1};

    nda::array<dcomplex,2> F_up_block0 = {{0}};
    nda::array<dcomplex,2> F_up_block1 = {{1, 0}};
    nda::array<dcomplex,2> F_up_block2 = {{0}, {1}};
    std::vector<nda::array<dcomplex,2>> F_up_blocks = 
        {F_up_block0, F_up_block1, F_up_block2};
    BlockOp F_up(block_indices_F, F_up_blocks);

    nda::array<dcomplex,2> F_down_block0 = {{0}};
    nda::array<dcomplex,2> F_down_block1 = {{0, 1}};
    nda::array<dcomplex,2> F_down_block2 = {{-1}, {0}};
    std::vector<nda::array<dcomplex,2>> F_down_blocks = 
        {F_down_block0, F_down_block1, F_down_block2};
    BlockOp F_down(block_indices_F, F_down_blocks);

    std::vector<BlockOp> Fs = {F_up, F_down};

    BlockDiagOpFun OCA_result = OCA_bs(hyb, hyb, itops, beta, Gt, Fs);
}

TEST(BlockSparseOCATest, OCA_single_exponential) {
    // DLR parameters
    double beta = 1.0;
    double Lambda = 100.0;
    double eps = 1.0e-13;
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);
    auto const & dlr_it = itops.get_itnodes();
    int r = itops.rank();

    auto dlr_it_abs = cppdlr::rel2abs(dlr_it);

    // create hybridization
    double D = 1.0;
    auto Deltat = nda::array<dcomplex,3>(r,1,1);
    auto Deltat_refl = nda::array<dcomplex,3>(r,1,1);
    Deltat(_,0,0) = exp(-D*dlr_it_abs*beta);
    Deltat_refl(_,0,0) = exp(D*dlr_it_abs*beta);

    // create Green's function
    double g = -1.0;
    auto Gt_block = nda::array<dcomplex,3>(r,1,1);
    auto Gt_zero_block_index = nda::ones<int>(1);
    Gt_block(_,0,0) = exp(-g*dlr_it_abs*beta);
    std::vector<nda::array<dcomplex,3>> Gt_blocks = {Gt_block};
    auto Gt = BlockDiagOpFun(Gt_blocks, Gt_zero_block_index);

    // create annihilation operator
    auto F_block = nda::ones<dcomplex>(1,1);
    auto F_block_indices = nda::vector<int>(1);
    F_block_indices = 0;
    std::vector<nda::array<dcomplex,2>> F_blocks = {F_block};
    auto F = BlockOp(F_block_indices, F_blocks);
    std::vector<BlockOp> Fs = {F};

    auto OCA_result = OCA_bs(Deltat, Deltat_refl, itops, beta, Gt, Fs);
    auto OCA_ana = nda::zeros<dcomplex>(r);
    for (int i = 0; i < r; i++) {
        auto tau = dlr_it_abs(i);
        OCA_ana(i) = (1 - exp(-tau) - tau) - exp(2*tau)*(-1 + exp(tau) - tau) + (-1 + exp(tau))*(-1 + exp(tau));
    }

    EXPECT_LT(nda::norm(OCA_result.get_block(0)(_,0,0)-OCA_ana), 1.0e-10);
}

/*
TEST(BlockSparseXCATest, load_BlockOp) {
    std::string fname = "/home/paco/feynman/ppsc-soe/benchmarks/atom_diag_eval/atom_diag_to_text.txt";

    std::cout << " " << std::endl;
    std::cout << "print from text" << std::endl;
    text2BlockOp(fname);
}
*/

TEST(BlockSparseXCATest, load_hdf5) {
    h5::file hfile("/home/paco/feynman/ppsc-soe/test/c++/two_band_ad.h5", 'r');
    
    h5::group hgroup(hfile);
    h5::group ad = hgroup.open_group("ad");

    nda::array<long, 2> ann_conn(4,14);
    h5::read(ad, "annihilation_connection", ann_conn);
    std::cout << ann_conn << std::endl;
}

TEST(BlockSparseOCATest, twoband) {
    // DLR parameters
    double beta = 2.0;
    double Lambda = 10*beta;
    double eps = 1.0e-10;
    // DLR generation
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);
    auto const & dlr_it = itops.get_itnodes();
    int r = itops.rank();

    // hybridization parameters
    double s = 0.5;
    double t = 1.0;
    nda::array<double,1> e{-2.3*t, 2.3*t};
    // hybridization generation
    auto Jt = nda::array<dcomplex,3>(r,1,1);
    auto Jt_refl = nda::array<dcomplex,3>(r,1,1);
    for (int i = 0; i <= 1; i++) {
        Jt(_,0,0) += exp(-e(i)*dlr_it)/(1 + exp(-e(i)*beta));
        Jt_refl(_,0,0) += exp(e(i)*dlr_it)/(1 + exp(-e(i)*beta));
    }
    auto Deltat = nda::array<dcomplex,3>(r,2,2);
    auto Deltat_refl = nda::array<dcomplex,3>(r,2,2);
    Deltat(_,0,0) = Jt(_,0,0);
    Deltat(_,1,1) = Jt(_,0,0);
    Deltat(_,0,1) = -s*Jt(_,0,0);
    Deltat(_,1,0) = -s*Jt(_,0,0);
    Deltat = t*t*Deltat;
    Deltat_refl(_,0,0) = Jt_refl(_,0,0);
    Deltat_refl(_,1,1) = Jt_refl(_,0,0);
    Deltat_refl(_,0,1) = -s*Jt_refl(_,0,0);
    Deltat_refl(_,1,0) = -s*Jt_refl(_,0,0);
    Deltat_refl = t*t*Deltat_refl;

    // use src/dlr_dyson_ppsc.hpp/free_gf_ppsc to get nonint. Green's function
    // from Hamiltonian
    // 
    // TODO: read from hdf5 file
    //
    // parameters
    double U = 2.0;
    double J_H = 0.2;

    auto H_loc = nda::zeros<dcomplex>(16,16);
    H_loc(5,5) = 1.4;
    H_loc(6,6) = 1.4;
    H_loc(7,7) = 1.6;
    H_loc(8,8) = 1.6;
    H_loc(9,9) = 2.0;
    H_loc(9,10) = 0.2;
    H_loc(10,9) = 0.2;
    H_loc(10,10) = 2.0;
    H_loc(11,11) = 5.0;
    H_loc(12,12) = 5.0;
    H_loc(13,13) = 5.0;
    H_loc(14,14) = 5.0;
    H_loc(15,15) = 10.0;

    auto H_loc_eig = nda::linalg::eigenelements(H_loc);
    dcomplex tr_exp_minusbetaH = 0;
    for (int i = 0; i < 16; i++) {
        tr_exp_minusbetaH += exp(-beta*std::get<0>(H_loc_eig)(i));
    }

    auto eta_0 = nda::log(-1 / tr_exp_minusbetaH) / beta;
    
    auto Gt_evals_t = nda::zeros<dcomplex>(16, 16); 
    auto Gt_mat = nda::zeros<dcomplex>(r, 16, 16);
    auto Gbeta = nda::zeros<dcomplex>(16, 16);
    for (int t = 0; t < r; t++) {
        for (int i = 0; i < 16; i++) {
            Gt_evals_t(i,i) = exp(-dlr_it(t)*(std::get<0>(H_loc_eig)(i) - eta_0));
        }
        Gt_mat(t,_,_) = nda::matmul(
            std::get<1>(H_loc_eig), 
            nda::matmul(Gt_evals_t, nda::transpose(std::get<1>(H_loc_eig))));
    }
    for (int i = 0; i < 16; i++) {
        Gbeta(i,i) = exp(-beta*(std::get<0>(H_loc_eig)(i) - eta_0));
    }
    Gbeta = nda::matmul(Gbeta, nda::transpose(std::get<1>(H_loc_eig)));
    Gbeta = nda::matmul(std::get<1>(H_loc_eig), Gbeta);
    ASSERT_LE(nda::abs(nda::trace(Gbeta) + 1), 1e-10);

    // TODO
    // create Gt BlockDiagOpFun
    // get Fops
}

TEST(BlockSparseBackboneTest, OCA) {
    double beta = 100;
    double eps = 1e-10;
    double Lambda = 100;
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);

    int N = 4;
    int r = dlr_rf.size();
    int n = 2;

    // set up hybridization
    nda::array<dcomplex,3> hyb({r, n, n});
    for (int t = 0; t < r; ++t) {
        hyb(t,0,0) = 1;
        hyb(t,1,1) = -1;
        hyb(t,0,1) = -1;
        hyb(t,1,0) = 4;
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
    nda::vector<int> zero_block_indices = {-1, 0, 0};
    BlockDiagOpFun Gt(Gt_blocks, zero_block_indices);

    // set up annihilation operators
    nda::vector<int> block_indices_F = {-1, 0, 1};

    nda::array<dcomplex,2> F_up_block0 = {{0}};
    nda::array<dcomplex,2> F_up_block1 = {{1, 0}};
    nda::array<dcomplex,2> F_up_block2 = {{0}, {1}};
    std::vector<nda::array<dcomplex,2>> F_up_blocks = 
        {F_up_block0, F_up_block1, F_up_block2};
    BlockOp F_up(block_indices_F, F_up_blocks);

    nda::array<dcomplex,2> F_down_block0 = {{0}};
    nda::array<dcomplex,2> F_down_block1 = {{0, 1}};
    nda::array<dcomplex,2> F_down_block2 = {{-1}, {0}};
    std::vector<nda::array<dcomplex,2>> F_down_blocks = 
        {F_down_block0, F_down_block1, F_down_block2};
    BlockOp F_down(block_indices_F, F_down_blocks);

    auto F_up_dag = dagger_bs(F_up);
    auto F_down_dag = dagger_bs(F_down);

    // TODO: abstract representation for backbone diagrams?
}