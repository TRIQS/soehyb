#include <cppdlr/dlr_imfreq.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <h5/complex.hpp>
#include <h5/object.hpp>
#include <ios>
#include <limits>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/mapped_functions.hxx>
#include <nda/nda.hpp>
#include <block_sparse.hpp>
#include <gtest/gtest.h>
#include <iomanip>

using namespace nda;

nda::array<dcomplex,3> Hmat_to_Gtmat(nda::array<dcomplex,2> Hmat, double beta, nda::array<double,1> dlr_it_abs) {
    int N = Hmat.extent(0);
     auto [H_loc_eval, H_loc_evec] = nda::linalg::eigenelements(Hmat);
    auto E0 = nda::min_element(H_loc_eval);
    H_loc_eval -= E0;
    auto tr_exp_minusbetaH = nda::sum(exp(-beta*H_loc_eval));
    auto eta_0 = nda::log(tr_exp_minusbetaH) / beta;
    H_loc_eval += eta_0;
    auto Gt_evals_t = nda::zeros<dcomplex>(N, N); 
    int r = dlr_it_abs.extent(0);
    auto Gt_mat = nda::zeros<dcomplex>(r, N, N);
    auto Gbeta = nda::zeros<dcomplex>(N, N);
    for (int t = 0; t < r; t++) {
        for (int i = 0; i < N; i++) {
            Gt_evals_t(i,i) = -exp(-beta*dlr_it_abs(t)*H_loc_eval(i));
        }
        Gt_mat(t,_,_) = nda::matmul(
            H_loc_evec, 
            nda::matmul(Gt_evals_t, nda::transpose(H_loc_evec)));
    }
    return Gt_mat;
}

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
        NCA_result_dense(t,_,_) -=
            hyb(0,0,0)*nda::matmul(temp_dense, F_up_dag_dense);
        temp_dense = nda::matmul(F_up_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= 
            hyb(0,0,1)*nda::matmul(temp_dense, F_down_dag_dense);
        temp_dense = nda::matmul(F_down_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= 
            hyb(0,1,0)*nda::matmul(temp_dense, F_up_dag_dense);
        temp_dense = nda::matmul(F_down_dense, Gt_dense(t,_,_));
        NCA_result_dense(t,_,_) -= 
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
        NCA_result_dense(t,_,_) -= 
            hyb(0,1,1)*nda::matmul(temp_dense, F_down_dense);
    }

    EXPECT_EQ(NCA_result.get_block(0)(_,0,0), NCA_result_dense(_,0,0));
    EXPECT_EQ(NCA_result.get_block(1), NCA_result_dense(_,range(1,3),range(1,3)));
    EXPECT_EQ(NCA_result.get_block(2)(_,0,0), NCA_result_dense(_,3,3));
}

// TODO: another NCA test
TEST(BlockSparseNCATest, single_exponential) {
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
    double D = 0.03;
    auto Deltat = nda::array<dcomplex,3>(r,1,1);
    auto Deltat_refl = nda::array<dcomplex,3>(r,1,1);
    Deltat(_,0,0) = exp(-D*dlr_it_abs*beta);
    Deltat_refl(_,0,0) = exp(D*dlr_it_abs*beta);

    // create Green's function
    double g = -0.54;
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
        // NCA_ana(i) = exp(2*tau) - 1;
        NCA_ana(i) = -exp(-(D+g)*tau) - exp((D-g)*tau);
    }

    EXPECT_LT(
        nda::norm(
            (NCA_result.get_block(0)(_,0,0)-NCA_ana), 
            std::numeric_limits<double>::infinity())
        /nda::norm(
            NCA_ana, 
            std::numeric_limits<double>::infinity()), 
        1.0e-12);
}

TEST(BlockSparseOCATest, single_exponential) {
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
    double D = 1.0;//0.07;
    auto Deltat = nda::array<dcomplex,3>(r,1,1);
    auto Deltat_refl = nda::array<dcomplex,3>(r,1,1);
    Deltat(_,0,0) = exp(-D*dlr_it_abs*beta);
    Deltat_refl(_,0,0) = exp(D*dlr_it_abs*beta);

    // create Green's function
    double g = -1.0;//-0.0023;
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

    auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);
    auto OCA_ana = nda::zeros<dcomplex>(r);
    auto OCA_ana_fb10 = nda::zeros<dcomplex>(r);
    for (int i = 0; i < r; i++) {
        auto tau = dlr_it_abs(i);
        // OCA_ana(i) = (1 - exp(-tau) - tau) - exp(2*tau)*(-1 + exp(tau) - tau) + (-1 + exp(tau))*(-1 + exp(tau));
        // after sign fix v
        // OCA_ana(i) = -exp(-2 - tau) * (exp(1) + exp(tau)) 
        //     * (exp(1) - exp(2*tau) + exp(3*tau) - exp(1+tau) - exp(2*tau)*tau + exp(1+tau)*tau);
        OCA_ana(i) = -exp(-g*tau - 2*D*(1+tau)) * (exp(D) + exp(D*tau)) 
            * (exp(D) + exp(3*D*tau) + exp(D*(1+tau)) * (-1 + D*tau) 
            - exp(2*D*tau) * (1 + D*tau)) / (D*D);
        OCA_ana_fb10(i) = exp(-D - g*tau) * (-1 + nda::cosh(D*tau)) / (D * D);
        // OCA_ana_fb10(i) = exp(-(D + g)*tau) * pow((-1 + exp(D*tau)), 2) / (2 * D * D);
    }
    EXPECT_LT(
        nda::norm(
            (OCA_result.get_block(0)(_,0,0)-OCA_ana), 
            std::numeric_limits<double>::infinity())/
        nda::norm(OCA_ana, std::numeric_limits<double>::infinity()), 
        1.0e-12);
    nda::array<dcomplex,3> Fs_dense = {{{1.0}}};
    auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_block, Fs_dense, Fs_dense);

    nda::array<int,3> h5_test = {{{3, 4}, {2, 3}}, {{1, 2}, {2, 5}}};

    h5::file hfile("/home/paco/feynman/saved_data/OCA_single_exponential_test.h5", 'w');

    h5::array_interface::array_view OCA_result_view(h5::hdf5_type<dcomplex>(), (void *) OCA_result.get_block(0).data(), 1, true);
    h5::array_interface::write(hfile, "OCA_result", OCA_result_view, false);

    h5::array_interface::array_view OCA_dense_result_view(h5::hdf5_type<dcomplex>(), OCA_dense_result.data(), 3, true);
    h5::array_interface::write(hfile, "OCA_dense_result", OCA_dense_result_view, false);

    h5::array_interface::array_view OCA_ana_fb10_view(h5::hdf5_type<dcomplex>(),  OCA_ana_fb10.data(), 1, true);
    h5::array_interface::write(hfile, "OCA_ana_fb10", OCA_ana_fb10_view, false);

    h5::array_interface::array_view view(h5::hdf5_type<int>(), (void*) h5_test.data(), 3, false);
    view.slab.count = {2, 2, 2};
    view.parent_shape = {2, 2, 2};
    h5::array_interface::write(hfile, "view", view, false);

    std::cout << "OCA_result = " << OCA_result << std::endl;
    std::cout << "OCA_dense_result = " << OCA_dense_result << std::endl;
    std::cout << "OCA_ana_fb10 = " << OCA_ana_fb10 << std::endl;

    int n_quad = 100;
    auto OCA_tpz_result = OCA_tpz(Deltat, itops, beta, Gt_block, Fs_dense, n_quad);
    auto OCA_dense_result_coeffs = itops.vals2coefs(OCA_dense_result);
    auto it_eq = cppdlr::eqptsrel(n_quad+1);
    auto OCA_dense_result_eq = nda::array<dcomplex,3>(n_quad+1,1,1);
    for (int i = 0; i <= n_quad; i++) {
        OCA_dense_result_eq(i,_,_) = itops.coefs2eval(OCA_dense_result_coeffs, it_eq(i));
    }

    std::cout << std::endl;
    std::cout << "OCA dense on eq grid = " << OCA_dense_result_eq(_,0,0) << std::endl;
    std::cout << std::endl;
    std::cout << "OCA tpz result = " << OCA_tpz_result(_,0,0) << std::endl;
}

TEST(BlockSparseMisc, load_hdf5) {
    h5::file hfile("/home/paco/feynman/ppsc-soe/benchmarks/atom_diag_eval/two_band_ad.h5", 'r');
    
    h5::group hgroup(hfile);
    h5::group ad = hgroup.open_group("ad");

    long num_blocks;
    h5::read(hgroup, "num_blocks", num_blocks);

    nda::array<long, 2> ann_conn(4,num_blocks);
    nda::array<long, 2> cre_conn(4,num_blocks);
    h5::read(ad, "annihilation_connection", ann_conn);
    h5::read(ad, "creation_connection", cre_conn);

    std::vector<nda::array<double,2>> H_blocks;
    h5::read(hgroup, "H_mat_blocks", H_blocks);

    std::vector<nda::array<double,2>> dummy(14);
    std::vector<std::vector<nda::array<double,2>>> F_blocks(4, dummy);
    std::vector<std::vector<nda::array<double,2>>> Fdag_blocks(4, dummy);
    
    for (int i = 0; i < 4; i++) {
        h5::read(hgroup, "c_blocks/" + std::to_string(i), F_blocks[i]);
        h5::read(hgroup, "cdag_blocks/" + std::to_string(i), Fdag_blocks[i]);
    }
}

TEST(BlockSparseMisc, compute_nonint_gf) {
    // DLR parameters
    double beta = 2.0;
    double Lambda = 10*beta;
    double eps = 1.0e-10;
    // DLR generation
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);
    auto const & dlr_it = itops.get_itnodes();
    auto dlr_it_abs = cppdlr::rel2abs(dlr_it);
    int r = itops.rank();

    // get Hamiltonian from hdf5 file
    h5::file hfile("/home/paco/feynman/ppsc-soe/benchmarks/atom_diag_eval/two_band_ad.h5", 'r');
    h5::group hgroup(hfile);
    h5::group ad = hgroup.open_group("ad");
    long num_blocks;
    h5::read(hgroup, "num_blocks", num_blocks);
    std::vector<nda::array<double,2>> H_blocks; // blocks of Hamiltonian
    h5::read(hgroup, "H_mat_blocks", H_blocks);
    nda::array<long,1> H_block_inds(num_blocks); // block col indices of Hamiltonian
    h5::read(hgroup, "H_mat_block_inds", H_block_inds);
    auto H_dense = nda::zeros<dcomplex>(16,16);
    h5::read(hgroup, "H_mat_dense", H_dense); // Hamiltonian in dense storage
    std::cout << H_dense << std::endl;

    h5::file hfile2("/home/paco/feynman/ppsc-soe/benchmarks/twoband/saved/G0_iaa_beta=2.0.h5", 'r');
    h5::group hgroup2(hfile2);
    auto G0_py = nda::zeros<dcomplex>(r,16,16);
    h5::read(hgroup2, "G0_iaa", G0_py);

    // compute noninteracting Green's function from dense Hamiltonian
    auto [H_loc_eval, H_loc_evec] = nda::linalg::eigenelements(H_dense);
    auto E0 = nda::min_element(H_loc_eval);
    H_loc_eval -= E0;
    auto tr_exp_minusbetaH = nda::sum(exp(-beta*H_loc_eval));
    auto eta_0 = nda::log(tr_exp_minusbetaH) / beta;
    H_loc_eval += eta_0;
    auto Gt_evals_t = nda::zeros<dcomplex>(16, 16); 
    auto Gt_mat = nda::zeros<dcomplex>(r, 16, 16);
    auto Gbeta = nda::zeros<dcomplex>(16, 16);
    Gt_mat = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
    for (int i = 0; i < 16; i++) {
        Gbeta(i,i) = -exp(-beta*H_loc_eval(i));
    }
    Gbeta = nda::matmul(Gbeta, nda::transpose(H_loc_evec));
    Gbeta = nda::matmul(H_loc_evec, Gbeta);
    // check that trace of noninteracting Green's function from dense 
    // Hamiltonian at tau = beta has trace 1
    ASSERT_LE(nda::abs(nda::trace(Gbeta) + 1), 1e-13);

    auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);
    // check that the noninteracting Green's function, computing from the 
    // sparse- and dense-storage Hamiltonians are the same
    ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_,range(0,4),range(0,4)) - Gt.get_block(0))), 1e-13);
    ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_,range(4,10),range(4,10)) - Gt.get_block(1))), 1e-13);
    ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_,range(10,11),range(10,11)) - Gt.get_block(2))), 1e-13);
    ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_,range(11,15),range(11,15)) - Gt.get_block(3))), 1e-13);
    ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_,range(15,16),range(15,16)) - Gt.get_block(4))), 1e-13);

    std::cout << G0_py(0,range(4,10),range(4,10)) << std::endl;
    std::cout << Gt_mat(0,range(4,10),range(4,10)) << std::endl;
    ASSERT_LE(nda::max_element(nda::abs(G0_py - Gt_mat)), 1e-13);
}

TEST(BlockSparseOCATest, two_band_discrete_bath) {
    // DLR parameters
    double beta = 1.0;
    double Lambda = 1000*beta;
    double eps = 1.0e-10;
    // DLR generation
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);
    auto const & dlr_it = itops.get_itnodes();
    auto dlr_it_abs = cppdlr::rel2abs(dlr_it);
    int r = itops.rank();

    // hybridization parameters
    double s = 0.5;
    double t = 1.0;
    nda::array<double,1> e{-2.3*t, 2.3*t};
    
    // hybridization generation
    auto Jt = nda::array<dcomplex,3>(r,1,1);
    auto Jt_refl = nda::array<dcomplex,3>(r,1,1);
    for (int i = 0; i <= 1; i++) {
        for (int t = 0; t < r; t++) {
            Jt(t,0,0) += k_it(dlr_it_abs(t), e(i), beta);
            Jt_refl(t,0,0) += k_it(-dlr_it_abs(t), e(i), beta);
        }
    }
    // orbital index order: do 0, do 1, up 0, up 1. same level <-> same parity index
    // TODO: read this info from hdf5 file, ad/h_atomic/fundamental_operator_set
    auto Deltat = nda::array<dcomplex,3>(r,4,4);
    auto Deltat_refl = nda::array<dcomplex,3>(r,4,4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                Deltat(_,i,j) = Jt(_,0,0);
                Deltat_refl(_,i,j) = Jt_refl(_,0,0);
            }
            else if (i == 0 && j == 1 || i == 1 && j == 0 || i == 2 && j == 3 || i == 3 && j == 2) {
                Deltat(_,i,j) = s*Jt(_,0,0);
                Deltat_refl(_,i,j) = s*Jt_refl(_,0,0);
            }
        }
    }
    Deltat = t*t*Deltat;
    Deltat_refl = t*t*Deltat_refl;

    // get Hamiltonian, creation/annihilation operators in block-sparse storage
    h5::file hfile("/home/paco/feynman/ppsc-soe/benchmarks/atom_diag_eval/two_band_ad.h5", 'r');
    h5::group hgroup(hfile);
    h5::group ad = hgroup.open_group("ad");
    long num_blocks;
    h5::read(hgroup, "num_blocks", num_blocks);

    // Hamiltonian
    std::vector<nda::array<double,2>> H_blocks;
    h5::read(hgroup, "H_mat_blocks", H_blocks);
    nda::array<long,1> H_block_inds(num_blocks);
    h5::read(hgroup, "H_mat_block_inds", H_block_inds);

    // Green's function
    auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);

    // creation/annihilation operators
    nda::array<long,2> ann_conn(4,num_blocks);
    nda::array<long,2> cre_conn(4,num_blocks);
    h5::read(ad, "annihilation_connection", ann_conn); // block column indices of F operators
    h5::read(ad, "creation_connection", cre_conn); // block column indices of Fdag operators
    std::vector<BlockOp> Fs;
    std::vector<BlockOp> Fdags;
    std::vector<nda::array<dcomplex,2>> dummy(num_blocks);
    std::vector<std::vector<nda::array<dcomplex,2>>> F_blocks(4, dummy);
    std::vector<std::vector<nda::array<dcomplex,2>>> Fdag_blocks(4, dummy);
    for (int i = 0; i < 4; i++) {
        h5::read(hgroup, "c_blocks/" + std::to_string(i), F_blocks[i]);
        h5::read(hgroup, "cdag_blocks/" + std::to_string(i), Fdag_blocks[i]);
    }
    for (int i = 0; i < 4; i++) {
        nda::vector<int> F_block_indices = ann_conn(i,_);
        Fs.emplace_back(BlockOp(F_block_indices, F_blocks[i]));
        nda::vector<int> Fdag_block_indices = cre_conn(i,_);
        Fdags.emplace_back(BlockOp(Fdag_block_indices, Fdag_blocks[i]));
    }

    // subspace indices
    std::vector<unsigned long> dummy2;
    std::vector<std::vector<unsigned long>> subspaces(num_blocks, dummy2);
    for (int i = 0; i < num_blocks; i++) {
        h5::read(ad, "sub_hilbert_spaces/" + std::to_string(i) + "/fock_states", subspaces[i]);
    }
    std::vector<long> fock_state_order(begin(subspaces[0]), end(subspaces[0]));
    for (int i = 1; i < num_blocks; i++) {
        fock_state_order.insert(end(fock_state_order), begin(subspaces[i]), end(subspaces[i]));
    }

    // Hamiltonian, creation/annihilation operators in dense storage
    // Hamiltonian 
    auto H_dense = nda::zeros<dcomplex>(16,16);
    h5::read(hgroup, "H_mat_dense", H_dense);

    // Green's function 
    auto Gt_dense = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);

    // creation/annihilation operators
    auto Fs_dense = nda::zeros<dcomplex>(4,16,16);
    h5::read(hgroup, "c_dense", Fs_dense);
    auto F_dags_dense = nda::zeros<dcomplex>(4,16,16);
    for (int i = 0; i < 4; i++) {
        F_dags_dense(i,_,_) = nda::transpose(nda::conj(Fs_dense(i,_,_)));
    }

    // block-sparse NCA and OCA compuations
    auto NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);
    auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);
    /*
    for (int i = 0; i < 1; i++) {
        std::cout << "OCA = " << OCA_result.get_block(0)(_,i,i) << std::endl;
    }*/

    // load NCA and OCA results from twoband.py
    std::string Lambda_str = (Lambda == 10.0) ? "10.0" : (Lambda == 100.0) ? "100.0" : "1000.0";
    h5::file Gtfile("/home/paco/feynman/ppsc-soe/benchmarks/twoband/saved/G0_iaa_beta=1.0_Lambda=" + Lambda_str + ".h5", 'r');
    h5::group Gtgroup(Gtfile);
    auto NCA_py = nda::zeros<dcomplex>(r,16,16);
    h5::read(Gtgroup, "NCA", NCA_py);
    auto OCA_py = nda::zeros<dcomplex>(r,16,16);
    h5::read(Gtgroup, "OCA", OCA_py);

    auto NCA_dense_result = NCA_dense(Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense);
    auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

    auto NCA_py_perm = nda::zeros<dcomplex>(r,16,16);
    auto OCA_py_perm = nda::zeros<dcomplex>(r,16,16);
    for (int t = 0; t < r; t++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                NCA_py_perm(t,i,j) = NCA_py(t,fock_state_order[i],fock_state_order[j]);
                OCA_py_perm(t,i,j) = OCA_py(t,fock_state_order[i],fock_state_order[j]);
            }
        }
    }

    // check that dense NCA and OCA calculations agree with twoband.py
    ASSERT_LE(nda::max_element(nda::abs(NCA_dense_result - NCA_py_perm)), eps);
    ASSERT_LE(nda::max_element(nda::abs(OCA_dense_result - OCA_py_perm + NCA_py_perm)), eps);

    // check that block-sparse NCA and OCA calculations agree with twoband.py
    int s0 = 0; 
    int s1 = subspaces[0].size();
    for (int i = 0; i < num_blocks; i++) { // compare each block
        ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(i) - NCA_py_perm(_,range(s0,s1),range(s0,s1)))), eps);
        ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - OCA_py_perm(_,range(s0,s1),range(s0,s1)) + NCA_py_perm(_,range(s0,s1),range(s0,s1)))), eps);
        s0 = s1;
        if (i < num_blocks - 1) s1 += subspaces[i+1].size();
    }
    
}

TEST(BlockSparseOCATest, two_band_semicircle_bath) {
    double beta = 8.0;
    double Lambda = 10*beta;
    double eps = 1.0e-6;

    // DLR generation
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);
    auto const & dlr_it = itops.get_itnodes();
    auto dlr_it_abs = cppdlr::rel2abs(dlr_it);
    int r = itops.rank();

    // hybridization parameters
    double s = 1.0;
    double t = 1.0;
    double D = 2*t;
    
    // hybridization generation
    auto Deltat = nda::array<dcomplex,3>(r,4,4);
    auto Deltat_refl = nda::array<dcomplex,3>(r,4,4);
    h5::file delta_h5("/home/paco/feynman/ppsc-soe/benchmarks/twoband/saved/Delta_bethe.h5", 'r');
    h5::group delta_group(delta_h5);
    h5::read(delta_group, "hyb", Deltat);

    Deltat_refl = itops.reflect(Deltat);

    // get Hamiltonian, creation/annihilation operators from hdf5 file
    h5::file hfile("/home/paco/feynman/ppsc-soe/benchmarks/atom_diag_eval/two_band_ad.h5", 'r');
    h5::group hgroup(hfile);
    h5::group ad = hgroup.open_group("ad");
    long num_blocks;
    h5::read(hgroup, "num_blocks", num_blocks);
    nda::array<long,2> ann_conn(4,num_blocks);
    nda::array<long,2> cre_conn(4,num_blocks);
    h5::read(ad, "annihilation_connection", ann_conn); // block column indices of F operators
    h5::read(ad, "creation_connection", cre_conn); // block column indices of Fdag operators
    std::vector<nda::array<double,2>> H_blocks;
    h5::read(hgroup, "H_mat_blocks", H_blocks);
    nda::array<long,1> H_block_inds(num_blocks);
    h5::read(hgroup, "H_mat_block_inds", H_block_inds);
    auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);
    std::cout << "Gt" << Gt.get_block(0)(_,0,0) << std::endl;

    std::vector<nda::array<dcomplex,2>> dummy(num_blocks);
    std::vector<std::vector<nda::array<dcomplex,2>>> F_blocks(4, dummy);
    std::vector<std::vector<nda::array<dcomplex,2>>> Fdag_blocks(4, dummy);
    
    for (int i = 0; i < 4; i++) {
        h5::read(hgroup, "c_blocks/" + std::to_string(i), F_blocks[i]);
        h5::read(hgroup, "cdag_blocks/" + std::to_string(i), Fdag_blocks[i]);
    }

    std::vector<BlockOp> Fs;
    std::vector<BlockOp> Fdags;
    for (int i = 0; i < 4; i++) {
        nda::vector<int> F_block_indices = ann_conn(i,_);
        Fs.emplace_back(BlockOp(F_block_indices, F_blocks[i]));
        nda::vector<int> Fdag_block_indices = cre_conn(i,_);
        Fdags.emplace_back(BlockOp(Fdag_block_indices, Fdag_blocks[i]));
    }
    
    std::cout << "dlr_it_abs" << dlr_it_abs << std::endl;
    auto NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs); 
    for (int i = 0; i < 1; i++) {
        std::cout << "NCA = " << NCA_result.get_block(0)(_,i,i) << std::endl;
    }
    auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);
    for (int i = 0; i < 1; i++) {
        std::cout << "OCA = " << OCA_result.get_block(0)(_,i,i) << std::endl;
    }
    // TODO: get correct coefficients
}

TEST(BlockSparseXCATest, OCA) {
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