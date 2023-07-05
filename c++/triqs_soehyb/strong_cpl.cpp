#include "strong_cpl.hpp"


using namespace cppdlr;
using namespace nda;

hyb_decomp::hyb_decomp(nda::array_const_view<dcomplex,3> Delta_dlr, nda::vector_const_view<double> dlr_rf, double eps){
    // obtain dlr_rank and dim of Green's function
    int dlr_rank = Delta_dlr.shape(0);
    int dim = Delta_dlr.shape(1);
    // prepare for svd
    auto s_vec = nda::array<double, 1>(dim);
    auto U_local = nda::matrix<dcomplex,F_layout>(dim, dim);
    auto VT_local = nda::matrix<dcomplex,F_layout>(dim, dim);
    nda::array<dcomplex, 2, F_layout> a;

    int max_num_pole = dim*dlr_rank;
    auto U_all = nda::matrix<dcomplex>(dim,max_num_pole);
    auto V_all = nda::matrix<dcomplex>(max_num_pole,dim);
    auto w_all = nda::vector<double>(max_num_pole);

    //loop over all dlr frequencies, do svd, truncate stuff that are too small
    int P = 0;
    for (int i=0;i<dlr_rank; ++i){
        a = Delta_dlr(i,_,_);
        nda::lapack::gesvd(a,s_vec,U_local,VT_local);
        for (int d=0; d<dim; ++d){
            if (s_vec(d)>eps){
                w_all(P) = dlr_rf(i);
                U_all(_,P) = U_local(_,d)*sqrt(s_vec(d));
                V_all(P,_) = VT_local(d,_)*sqrt(s_vec(d));
                P +=1;
            }
        }
    }
    w = w_all(range(P));
    U = transpose(U_all(_,range(P)));
    V = V_all(range(P),_);
}