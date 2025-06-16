#include <cppdlr/utils.hpp>
#include <nda/algorithms.hpp>
#include <nda/basic_functions.hpp>
#include <nda/mapped_functions.hxx>
#include <nda/nda.hpp>
#include <h5/h5.hpp>
#include <cppdlr/cppdlr.hpp>
#include <chrono>
#include <iomanip>

using namespace nda;
using namespace cppdlr;

array<dcomplex,3> OCA_tpz(
    array_const_view<dcomplex,3> hyb,
    imtime_ops &itops, 
    double beta, 
    array_const_view<dcomplex, 3> Gt, 
    array_const_view<dcomplex, 3> Fs, 
    int n_quad) {

    vector_const_view<double> dlr_rf = itops.get_rfnodes();
    vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.extent(0);
    int N = Gt.extent(1);

    auto hyb_coeffs = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl = itops.reflect(hyb);
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl); 

    // get F^dagger operators
    int num_Fs = Fs.extent(0);
    array<dcomplex,3> F_dags(num_Fs, N, N);
    for (int i = 0; i < num_Fs; ++i) {
        F_dags(i,_,_) = transpose(conj(Fs(i,_,_)));
    }

    // get equispaced grid and evaluate functions on grid
    auto it_eq = eqptsrel(n_quad+1);
    array<dcomplex,3> hyb_eq(n_quad+1, num_Fs, num_Fs);
    array<dcomplex,3> hyb_refl_eq(n_quad+1, num_Fs, num_Fs);
    auto Gt_coeffs = itops.vals2coefs(Gt);
    array<dcomplex,3> Gt_eq(n_quad+1, N, N);
    // auto hyb_eq = itops.coefs2eval(hyb, it_eq);
    for (int i = 0; i < n_quad+1; i++) {
        hyb_eq(i,_,_) = itops.coefs2eval(hyb_coeffs, it_eq(i));
        hyb_refl_eq(i,_,_) = itops.coefs2eval(hyb_refl_coeffs, it_eq(i));
        hyb_refl_eq(i,_,_) = transpose(hyb_refl_eq(i,_,_));
        Gt_eq(i,_,_) = itops.coefs2eval(Gt_coeffs, it_eq(i));
    }
    array<dcomplex,3> Sigma_eq(n_quad+1,N,N);

    double dt = beta/n_quad;

    for (int fb1 = 0; fb1 <= 1; fb1++) {
        for (int fb2 = 0; fb2 <= 1; fb2++) {
            // fb = 1 for forward line, else = 0
            // fb1 corresponds with line from 0 to tau_2
            auto const &F1list = (fb1==1) ? Fs(_,_,_) : F_dags(_,_,_);
            auto const &F2list = (fb2==1) ? Fs(_,_,_) : F_dags(_,_,_);
            auto const &F3list = (fb1==1) ? F_dags(_,_,_) : Fs(_,_,_);
            auto const &F4list = (fb2==1) ? F_dags(_,_,_) : Fs(_,_,_);
            auto const &hyb1 = (fb1==1) ? hyb_eq(_,_,_) : hyb_refl_eq(_,_,_);
            auto const &hyb2 = (fb2==1) ? hyb_eq(_,_,_) : hyb_refl_eq(_,_,_);

            // sums over orbital indices
            for (int lam = 0; lam < num_Fs; lam++) {
                for (int nu = 0; nu < num_Fs; nu++) {
                    for (int mu = 0; mu < num_Fs; mu++) {
                        for (int kap = 0; kap < num_Fs; kap++) {

                            // quadrature 
                            for (int i = 1; i <= n_quad; i++) {
                                for (int i1 = 1; i1 <= i; i1++) {
                                    for (int i2 = 0; i2 <= i1; i2++) {
                                        double w = 1.0;
                                        if (i1 == i) w = w/2;
                                        if (i2 == 0 || i2 == i1) w = w/2;
                                        auto FGFGFGF = matmul(F4list(nu,_,_), 
                                            matmul(Gt_eq(i-i1,_,_), 
                                            matmul(F3list(mu,_,_), 
                                            matmul(Gt_eq(i1-i2,_,_), 
                                            matmul(F2list(lam,_,_), 
                                            matmul(Gt_eq(i2,_,_), F1list(kap,_,_)))))));

                                        Sigma_eq(i,_,_) += -1*w*hyb2(i-i2,nu,lam)*hyb1(i1,mu,kap)*FGFGFGF; 
                                        // ^ sign on each term is the same because cppdlr::reflect accounts for negative sign
                                    } // sum over i2
                                } // sum over i1 
                            } // sum over i
                            
                        } // sum over kappa
                    } // sum over mu
                } // sum over nu
            } // sum over lambda
        } // sum over fb2
    } // sum over fb1

    Sigma_eq = dt*dt*Sigma_eq;
    
    return Sigma_eq;
}

int main() {
    // Operator parameters
    int n = 4; // number of orbital indices
    int N = 16; // size of Green's function, cre/ann operators

    // DLR parameters
    double beta = 2.0;
    double Lambda = 1000*beta;
    double eps = 1.0e-10;
    // DLR generation
    auto dlr_rf = build_dlr_rf(Lambda, eps);
    auto itops = imtime_ops(Lambda, dlr_rf);
    auto const & dlr_it = itops.get_itnodes();
    auto dlr_it_abs = rel2abs(dlr_it);
    int r = itops.rank(); 

    // read G0, F, F_dag, NCA, OCA from ppsc-soe/benchmarks/twoband/twoband.by
    h5::file Gtfile("/home/paco/feynman/ppsc-soe/benchmarks/twoband/saved/G0_iaa_beta=2.0_Lambda=2000.0.h5", 'r');
    h5::group Gtgroup(Gtfile);
    auto Gt_dense = zeros<dcomplex>(r,N,N);
    h5::read(Gtgroup, "G0_iaa_no_perm", Gt_dense);
    // 30 May 2025 read F from twoband.py
    auto Fs_dense = zeros<dcomplex>(n,N,N);
    h5::read(Gtgroup, "F_no_perm", Fs_dense);
    auto F_dags_dense = zeros<dcomplex>(n,N,N);
    h5::read(Gtgroup, "F_dag_no_perm", F_dags_dense);
    auto Deltat = zeros<dcomplex>(r,n,n);
    h5::read(Gtgroup, "delta_iaa", Deltat); 
    auto NCA_py = zeros<dcomplex>(r,N,N);
    h5::read(Gtgroup, "NCA", NCA_py);
    auto OCA_py = zeros<dcomplex>(r,N,N);
    h5::read(Gtgroup, "OCA", OCA_py);
    // subtract off NCA
    OCA_py = OCA_py - NCA_py;
    std::cout << Deltat.shape() << std::endl;

    // compute OCA using trapezoidal rule
    int n_quad = 100;
    std::cout << "number of trapezoidal quadrature points: " << n_quad << "\n" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto OCA_tpz_result = OCA_tpz(Deltat, itops, beta, Gt_dense, Fs_dense, n_quad);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "time elapsed to do trapezoidal quadrature: " << duration.count() << " ms\n" << std::endl;

    h5::file tpz_file("/home/paco/feynman/saved_data/OCA_two_band/tpz_vs_python_tpz_result_nquad_" + std::to_string(n_quad) + ".h5", 'w');
    h5::write(tpz_file, "OCA_tpz_result", OCA_tpz_result);

    // evaluate OCA_py on uniform grid
    auto OCA_py_coefs = itops.vals2coefs(OCA_py);
    auto it_eq = eqptsrel(n_quad + 1);
    auto OCA_py_eq = zeros<dcomplex>(n_quad + 1, N, N);
    for (int i = 0; i <= n_quad; i++) {
        OCA_py_eq(i,_,_) = itops.coefs2eval(OCA_py_coefs, it_eq(i));
    }

    auto err_entry = make_regular(nda::abs(OCA_py_eq(_,4,4) - OCA_tpz_result(_,4,4)));

    std::cout << std::fixed << std::setprecision(10) << std::scientific << "err in (4,4)-entry wrt time = " << err_entry << "\n" << std::endl;
    std::cout << "L^inf[0,beta] err in (4,4)-entry = " << max_element(err_entry) << "\n" << std::endl;

    auto err_full = make_regular(nda::abs(OCA_py_eq - OCA_tpz_result));
    std::cout << "L^inf err = " << max_element(err_full) << "\n" << std::endl;

    // saved values, compress = False:
    // n_quad = 5   L^inf one entry = 7.1625007534e-03  L^inf = 9.0792215542e-03  time = 1.1 s
    // n_quad = 10  L^inf one entry = 1.7756127548e-03  L^inf = 2.3487887160e-03  time = 6 s
    // n_quad = 20  L^inf one entry = 4.4293584841e-04  L^inf = 6.7301063442e-04  time = 40 s
    // n_quad = 40  L^inf one entry = 1.1067303602e-04  L^inf = 2.8723963046e-04  time = 4.5 min
    // n_quad = 80  L^inf one entry = 2.7664445025e-04  L^inf = 1.9075833911e-04  time = 35 min

    // saved values, compress = True:
    // n_quad = 5   L^inf one entry = 7.1786547158e-03  L^inf = 9.0014940430e-03
    // n_quad = 10  L^inf one entry = 1.7917667172e-03  L^inf = 2.2710612048e-03
    // n_quad = 20  L^inf one entry = 4.5908981079e-04  L^inf = 5.7551812305e-04
    // n_quad = 40  L^inf one entry = 1.2682699840e-04  L^inf = 1.5245092006e-04
    // n_quad = 80  L^inf one entry = 7.8050566395e-05  L^inf = 9.1607140270e-05

    return 0;
}