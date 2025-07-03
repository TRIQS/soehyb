#include "triqs_soehyb/strong_cpl.hpp"
#include <cppdlr/dlr_imfreq.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <h5/complex.hpp>
#include <h5/object.hpp>
#include <nda/algorithms.hpp>
#include <nda/basic_functions.hpp>
#include <nda/declarations.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/mapped_functions.hxx>
#include <nda/nda.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_manual.hpp>
#include <triqs_soehyb/block_sparse_backbone.hpp>

nda::array<dcomplex,3> Hmat_to_Gtmat(nda::array<dcomplex,2> Hmat, double beta, nda::array<double,1> dlr_it_abs) {
    std::size_t N = Hmat.extent(0);
     auto [H_loc_eval, H_loc_evec] = nda::linalg::eigenelements(Hmat);
    auto E0 = nda::min_element(H_loc_eval);
    H_loc_eval -= E0;
    auto tr_exp_minusbetaH = nda::sum(exp(-beta*H_loc_eval));
    auto eta_0 = nda::log(tr_exp_minusbetaH) / beta;
    H_loc_eval += eta_0;
    auto Gt_evals_t = nda::zeros<dcomplex>(N, N); 
    std::size_t r = dlr_it_abs.extent(0);
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

std::tuple<int, 
    nda::array<dcomplex,3>, 
    nda::array<dcomplex,3>, 
    BlockDiagOpFun, 
    std::vector<BlockOp>, 
    nda::array<dcomplex,3>, 
    nda::array<dcomplex,3>, 
    nda::array<dcomplex,3>, 
    std::vector<std::vector<unsigned long>>, 
    std::vector<long>> 
    two_band_discrete_bath_helper(double beta, double Lambda, double eps) {

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
        for (int u = 0; u < r; u++) {
            Jt(u,0,0) += k_it(dlr_it_abs(u), e(i), beta);
            Jt_refl(u,0,0) += k_it(-dlr_it_abs(u), e(i), beta);
        }
    }
    // orbital index order: do 0, do 1, up 0, up 1. same level <-> same parity index
    auto Deltat = nda::array<dcomplex,3>(r,4,4);
    auto Deltat_refl = nda::array<dcomplex,3>(r,4,4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                Deltat(_,i,j) = Jt(_,0,0);
                Deltat_refl(_,i,j) = Jt_refl(_,0,0);
            }
            else if ((i == 0 && j == 1) || (i == 1 && j == 0) || (i == 2 && j == 3) || (i == 3 && j == 2)) {
                Deltat(_,i,j) = s*Jt(_,0,0);
                Deltat_refl(_,i,j) = s*Jt_refl(_,0,0);
            }
        }
    }
    Deltat = t*t*Deltat;
    Deltat_refl = t*t*Deltat_refl;

    // get Hamiltonian, creation/annihilation operators in block-sparse storage
    h5::file hfile("../test/c++/h5/two_band_ad.h5", 'r');
    h5::group hgroup(hfile);
    h5::group ad = hgroup.open_group("ad");
    int num_blocks = 5; // number of blocks of Hamiltonian 

    // Hamiltonian
    std::vector<nda::array<double,2>> H_blocks(num_blocks); // Hamiltonian in sparse storage
    H_blocks[0] = nda::make_regular(-1*nda::eye<double>(4));
    H_blocks[1] = {{-0.6, 0, 0, 0, 0, 0}, 
    {0, 8.27955e-19, 0, 0, 0.2, 0}, 
    {0, 0, -0.4, 0.2, 0, 0}, 
    {0, 0, 0.2, -0.4, 0, 0}, 
    {0, 0.2, 0, 0, 8.27955e-19, 0}, 
    {0, 0, 0, 0, 0, -0.6}};
    H_blocks[2] = {{0}}; 
    H_blocks[3] = nda::make_regular(2*nda::eye<double>(4)); 
    H_blocks[4] = {{6}}; 
    nda::vector<int> H_block_inds = {0, 0, -1, 0, 0}; 

    // Green's function
    auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);

    // creation/annihilation operators
    nda::array<int,2> ann_conn = {{2, 0, -1, 1, 3}, 
    {2, 0, -1, 1, 3}, 
    {2, 0, -1, 1, 3}, 
    {2, 0, -1, 1, 3}}; // block column indices of F operators 
    nda::array<int,2> cre_conn = {{1, 3, 0, 4, -1}, 
    {1, 3, 0, 4, -1}, 
    {1, 3, 0, 4, -1}, 
    {1, 3, 0, 4, -1}}; // block column indices of F^dag operators
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
        Fs.emplace_back(F_block_indices, F_blocks[i]);
        nda::vector<int> Fdag_block_indices = cre_conn(i,_);
        Fdags.emplace_back(Fdag_block_indices, Fdag_blocks[i]);
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

    return std::make_tuple(num_blocks, Deltat, Deltat_refl, Gt, Fs, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order);
}

nda::array<dcomplex, 4> OCA_ppsc(
    nda::array<dcomplex,3> Deltat, 
    nda::array<dcomplex,3> Deltat_refl, 
    imtime_ops& itops, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    double beta, 
    double eps, 
    nda::array<dcomplex, 3> Gt_dense, 
    nda::array<dcomplex, 3> Fs_dense, 
    nda::array<dcomplex, 3> F_dags_dense
) {
    int r = itops.rank(); 
    std::size_t N = Gt_dense.extent(1); 
    nda::array<int,2> D2 = {{0, 2}, {1, 3}}; 
    auto Deltadlr = itops.vals2coefs(Deltat);  //obtain dlr coefficient of Delta(t)     
    nda::vector<double> dlr_rf_reflect = -dlr_rf;
    nda::array<dcomplex,3> Deltadlr_reflect = Deltadlr*1.0;
    for (int i = 0; i < Deltadlr.shape(0); ++i) Deltadlr_reflect(i,_,_) = transpose(Deltadlr(i,_,_));
    auto Delta_decomp = hyb_decomp(Deltadlr,dlr_rf,eps); //decomposition of Delta(t) using DLR coefficient
    auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect,dlr_rf_reflect,eps); // decomposition of Delta(-t) using DLR coefficient
    int dim = Deltat.shape(1);
    hyb_F Delta_F(16, r, dim);
    hyb_F Delta_F_reflect(16, r, dim);
    Delta_F.update_inplace(Delta_decomp,dlr_rf, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
    Delta_F_reflect.update_inplace(Delta_decomp_reflect,dlr_rf_reflect, dlr_it, F_dags_dense, Fs_dense);
    auto fb = nda::vector<int>(2); fb(1) = 0;

    // this is both hybridization are forward
    auto OCA_forward_forward = Sigma_OCA_calc(Delta_F,   Deltat,  Deltat_refl, Gt_dense, itops, beta,  Fs_dense,  F_dags_dense,  false);
    OCA_forward_forward = nda::make_regular(-1 * OCA_forward_forward); 

    //this is result of Delta(t-t1) forward, which is the sum of Delta(t2,t0) being forward and backward
    auto OCA_forward  = Sigma_OCA_calc(Delta_F,Deltat,  Deltat_refl, Gt_dense, itops, beta,  Fs_dense,  F_dags_dense, true);
    // another way to calculate the same thing
    auto OCA_forward2 = Sigma_Diagram_calc(Delta_F,Delta_F_reflect, D2,  Deltat,  Deltat_refl, Gt_dense, itops, beta,  Fs_dense,  F_dags_dense,  fb, true);

    //Get Delta(t-t1) forward Delta(t2,t0) backward result via subtraction:
    auto OCA_forward_backward = -OCA_forward - OCA_forward_forward; 
    // auto OCA_forward_backward = -OCA_forward + OCA_forward_forward; 

    // Get Delta(t-t1) backward Delta(t2,t0) forward
    auto fb2 = nda::vector<int>(2); fb2(1) = 1;
    auto OCA_backward_forward = Sigma_Diagram_calc(Delta_F,Delta_F_reflect, D2,  Deltat,  Deltat_refl, Gt_dense, itops, beta,  Fs_dense,  F_dags_dense,  fb2, false);
    OCA_backward_forward = nda::make_regular(-1 * OCA_backward_forward); 

    //Get Delta(t-t1) backward Delta(t2,t0) backward, from subtraction
    auto OCA_backward = Sigma_Diagram_calc(Delta_F,Delta_F_reflect, D2,  Deltat,  Deltat_refl, Gt_dense, itops, beta,  Fs_dense,  F_dags_dense,  fb2, true);
    auto OCA_backward_backward = -OCA_backward - OCA_backward_forward;
    // auto OCA_backward_backward = -OCA_backward + OCA_backward_forward;

    nda::array<dcomplex,4> OCA_ff_fb_bf_bb_f_b(6,r,N,N); 
    OCA_ff_fb_bf_bb_f_b(0,_,_,_) = OCA_forward_forward; 
    OCA_ff_fb_bf_bb_f_b(1,_,_,_) = OCA_forward_backward; 
    OCA_ff_fb_bf_bb_f_b(2,_,_,_) = OCA_backward_forward; 
    OCA_ff_fb_bf_bb_f_b(3,_,_,_) = OCA_backward_backward; 
    OCA_ff_fb_bf_bb_f_b(4,_,_,_) = OCA_forward; 
    OCA_ff_fb_bf_bb_f_b(5,_,_,_) = OCA_backward; 

    return OCA_ff_fb_bf_bb_f_b; 
}

int main() {
    int n = 4, N = 16; 
    double beta = 2.0; 
    double Lambda = 100.0*beta; // 1000.0*beta; 
    double eps = 1.0e-10; 
    auto [num_blocks, 
        Deltat, 
        Deltat_refl, 
        Gt, 
        Fs, 
        Gt_dense, 
        Fs_dense, 
        F_dags_dense, 
        subspaces, 
        fock_state_order] = two_band_discrete_bath_helper(beta, Lambda, eps);
    // DLR generation
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);
    auto const & dlr_it = itops.get_itnodes();
    auto dlr_it_abs = cppdlr::rel2abs(dlr_it);
    int r = itops.rank();

    nda::array<dcomplex,3> T(r,N,N);
    for (int t = 0; t < r; t++) T(t,_,_) = nda::eye<dcomplex>(N); 

    // compute Fbars and Fdagbars
    auto hyb_coeffs = itops.vals2coefs(Deltat); // hybridization DLR coeffs
    auto hyb_refl = itops.reflect(Deltat); // nda::make_regular(-itops.reflect(Deltat));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
    auto Fset = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs); 

    // compute NCA and OCA manually and OCA using generic backbone evaluator
    nda::array<int,2> T_OCA = {{0, 2}, {1, 3}}; 
    auto B_OCA = BackboneSignature(T_OCA, n); 
    auto OCA_result_generic = eval_backbone_dense(B_OCA, beta, itops, Deltat, Gt_dense, Fs_dense, F_dags_dense); 
    auto NCA_result = NCA_dense(Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense); 
    auto OCA_result_manual = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

    // preallocate intermediate arrays
    nda::array<dcomplex, 3> Sigma_L_OCA(r, N, N), Tmu_OCA(r, N, N), GKt_OCA(r, N, N);
    nda::array<dcomplex, 3> OCA_result_generic_ff(r, N, N), OCA_result_generic_fb(r, N, N), OCA_result_generic_bf(r, N, N), OCA_result_generic_bb(r, N, N);
    nda::array<dcomplex, 4> Tkaps_OCA(n, r, N, N);
    nda::vector<int> states_OCA(4), pole_inds_OCA(1); 

    auto B_OCA_2 = BackboneSignature(T_OCA, n); 
    nda::vector<int> fb_OCA_vec{1,1}; 
    B_OCA_2.set_directions(fb_OCA_vec); 
    int sign = -1;  
    eval_backbone_fixed_lines_dense(
        B_OCA_2, beta, itops, Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense, Fset.F_dag_bars, Fset.F_bars_refl, 
        dlr_it, dlr_rf, T, GKt_OCA, Tkaps_OCA, Tmu_OCA, states_OCA, Sigma_L_OCA, 
        pole_inds_OCA, sign, OCA_result_generic_ff);
    fb_OCA_vec(0) = 0; 
    B_OCA_2.set_directions(fb_OCA_vec); 
    sign = -1; 
    eval_backbone_fixed_lines_dense(
        B_OCA_2, beta, itops, Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense, Fset.F_dag_bars, Fset.F_bars_refl, 
        dlr_it, dlr_rf, T, GKt_OCA, Tkaps_OCA, Tmu_OCA, states_OCA, Sigma_L_OCA, 
        pole_inds_OCA, sign, OCA_result_generic_fb);
    fb_OCA_vec(1) = 0; 
    B_OCA_2.set_directions(fb_OCA_vec); 
    sign = 1; 
    eval_backbone_fixed_lines_dense(
        B_OCA_2, beta, itops, Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense, Fset.F_dag_bars, Fset.F_bars_refl, 
        dlr_it, dlr_rf, T, GKt_OCA, Tkaps_OCA, Tmu_OCA, states_OCA, Sigma_L_OCA, 
        pole_inds_OCA, sign, OCA_result_generic_bb);
    fb_OCA_vec(0) = 1; 
    B_OCA_2.set_directions(fb_OCA_vec); 
    sign = 1; 
    eval_backbone_fixed_lines_dense(
        B_OCA_2, beta, itops, Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense, Fset.F_dag_bars, Fset.F_bars_refl, 
        dlr_it, dlr_rf, T, GKt_OCA, Tkaps_OCA, Tmu_OCA, states_OCA, Sigma_L_OCA, 
        pole_inds_OCA, sign, OCA_result_generic_bf);

    // compute OCA using Zhen's code
    auto OCA_Zhen_dir = OCA_ppsc(Deltat, Deltat_refl, itops, dlr_it, dlr_rf, beta, eps, Gt_dense, Fs_dense, F_dags_dense); 
    auto OCA_Zhen = nda::make_regular(OCA_Zhen_dir(0,_,_,_) + OCA_Zhen_dir(1,_,_,_) + OCA_Zhen_dir(2,_,_,_) + OCA_Zhen_dir(3,_,_,_)); 

    std::cout << "manual = " << OCA_result_manual(10,_,_) << std::endl;
    // std::cout << "generic = " << OCA_result_generic(10,_,_) << std::endl;
    std::cout << "generic ff" << OCA_result_generic_ff(10,_,_) << std::endl;
    std::cout << "generic fb" << OCA_result_generic_fb(10,_,_) << std::endl;
    std::cout << "generic bf" << OCA_result_generic_bf(10,_,_) << std::endl;
    std::cout << "generic bb" << OCA_result_generic_bb(10,_,_) << std::endl;
    std::cout << "Zhen ff = " << OCA_Zhen_dir(0,10,_,_) << std::endl;
    std::cout << "Zhen fb = " << OCA_Zhen_dir(1,10,_,_) << std::endl;
    std::cout << "Zhen bf = " << OCA_Zhen_dir(2,10,_,_) << std::endl;
    std::cout << "Zhen bb = " << OCA_Zhen_dir(3,10,_,_) << std::endl;
    std::cout << "Zhen f = " << OCA_Zhen_dir(4,10,_,_) << std::endl;
    std::cout << "Zhen b = " << OCA_Zhen_dir(5,10,_,_) << std::endl;

    return 0; 
}