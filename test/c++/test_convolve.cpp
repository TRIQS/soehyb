#include "triqs_soehyb/strong_cpl.hpp"
#include "nda/nda.hpp"
#include <cppdlr/cppdlr.hpp>
#include <gtest/gtest.h>
#include <nda/algorithms.hpp>
#include <nda/layout/policies.hpp>

using namespace cppdlr;
using namespace nda;

TEST(test_dlr, test_dlr_convolve) {
    double lambda = 1;
    double beta = 1;
    double eps = 1e-14;
    auto dlr_rf = build_dlr_rf(lambda, eps);
    // Get DLR imaginary time object
    auto itops = imtime_ops(lambda, dlr_rf);
    auto const &dlr_it = itops.get_itnodes();
    int r = itops.rank();

    auto f                  = nda::vector<dcomplex>(r);
    auto g                  = nda::vector<dcomplex>(r);

    for (int i = 0; i < r; ++i) { 
        if (dlr_it(i)>0) f(i) =  exp(-0.1*dlr_it(i)) ; else  f(i) =  exp(-0.1*(1+dlr_it(i)));
        };

    //for (int i = 0; i < r; ++i) { g(i) =   exp(-0.1*dlr_it(i)) ;};
    g=f;

    // Get DLR coefficients of f and g
    auto fc = itops.vals2coefs(f);
    auto gc = itops.vals2coefs(g);

    auto h = itops.convolve(beta, Fermion, fc, gc,TIME_ORDERED);
    std::cout<<"f is "<< f<<std::endl<<"g is "<<g<<std::endl<<"their convolution is" << h<<std::endl ;
    for (int i = 0; i < r; ++i) if (dlr_it(i)>0) std::cout<< f(i)*dlr_it(i) <<" ";else std::cout<< f(i)*(1+dlr_it(i)) <<" "; 
    // h = itops.convolve(beta, Fermion, fc, gc);
    // std::cout<<"f is "<< f<<std::endl<<"g is "<<g<<std::endl<<"their convolution is" << h ;


}
