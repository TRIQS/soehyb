#include <cppdlr/dlr_imfreq.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <h5/complex.hpp>
#include <h5/object.hpp>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/mapped_functions.hxx>
#include <nda/nda.hpp>
#include <block_sparse.hpp>
#include <chrono>

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

int main() {
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
    // std::cout << "starting NCA_bs" << std::endl;
    // auto NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);
    std::cout << "starting OCA_dense" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    // auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);
    auto OCA_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense); 
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "done." << std::endl;
    auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time = " << duration.count() << std::endl;
    
    return 0;
}