#include "strong_cpl.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>


using namespace cppdlr;
using namespace nda;

hyb_decomp::hyb_decomp(nda::array_const_view<dcomplex,3> Delta_dlr, nda::vector_const_view<double> dlr_rf,double eps, nda::array_const_view<dcomplex,3> Deltat,nda::vector_const_view<double> dlr_it, bool check){
    // obtain dlr_rank and dim of hyb function
    int dlr_rank = Delta_dlr.shape(0);
    int dim = Delta_dlr.shape(1);
    // prepare for svd
    auto s_vec = nda::array<double, 1>(dim);
    auto U_local = nda::matrix<dcomplex,F_layout>(dim,dim);
    auto VT_local = nda::matrix<dcomplex,F_layout>(dim,dim);
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
    if (check==true){
        int r =Delta_dlr.shape(0);

        std::cout<< "calculating error of the decomposition, this step could be slow since we have not made use of matrice multiplications"<<std::endl;
        auto Deltat_approx =nda::array<dcomplex,3>(r,dim,dim);

        Deltat_approx = 0;
        

        for (int a=0;a<dim;++a){
            for (int b =0 ;b<dim; ++b){
                for (int R = 0;  R<P;++R){
                    for (int k=0;k<r;++k) {
                        Deltat_approx(k,a,b) += k_it(dlr_it(k), w(R))*U(R,a)*V(R,b);
                    }
                }
            }
        }
        //std::cout<<Deltat(0,0,0)<<Deltat_approx(0,0,0);
        std::cout << "Max error in Delta(t): " << max_element(abs((Deltat - Deltat_approx))) << std::endl;
    }
}

hyb_F::hyb_F(hyb_decomp &hyb_decomp, nda::vector_const_view<double> dlr_rf, nda::vector_const_view<double> dlr_it, double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag){
    auto U = make_array_view(hyb_decomp.U);
    auto V = make_array_view(hyb_decomp.V);
    auto w = make_array_view(hyb_decomp.w); 
    if (F.shape(0)!= U.shape(1) ) throw std::runtime_error("F matrices not in the right format, or F matrices and hybridization do not match");
    if (F.shape(1)!= F.shape(2)) throw std::runtime_error("F are not square matrices");
    int N = F.shape(1);
    int vec_len = N*N;
    auto F_reshape = reshape(F,F.shape(0),vec_len);
    auto F_dag_reshape = reshape(F_dag,F_dag.shape(0),vec_len);

    // construct U_c, V_c:
    int P = V.shape(0);
    auto U_c = reshape(matmul(U,F_dag_reshape),P,N,N);
    auto V_c = reshape(matmul(U,F_reshape),P,N,N);
    
    int r = dlr_it.shape(0); 
    
    //Finally construct Utilde, Vtilde, and c
    nda::vector<double> c2(P);
    nda::array<dcomplex,4> U2_tilde(r,P,N,N);
    nda::array<dcomplex,4> V2_tilde(r,P,N,N);

    for (int R=0;R<P;++R){
       for (int k=0;k<r;++k){
            U2_tilde(k,R,_,_)= k_it(dlr_it(k),w(R))*U_c(R,_,_);
       }
       if (w(R)<0){
            for (int k=0;k<r;++k) V2_tilde(k,R,_,_) = k_it(dlr_it(k),-w(R))*V_c(R,_,_);
            c2(R) = 1/k_it(0,-w(R));
       }
       else {
            for (int k=0;k<r;++k) V2_tilde(k,R,_,_) = k_it(dlr_it(k),w(R))*V_c(R,_,_);
            c2(R) = 1/k_it(0,w(R)); 
       }
    }
    c = c2;
    U_tilde = U2_tilde;
    V_tilde = V2_tilde;
    w0 = hyb_decomp.w;
}

nda::array<dcomplex,3> Diagram_calc(hyb_F &hyb_F,nda::array_const_view<int,2> D,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Gt, nda::vector_const_view<double> dlr_it,nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag){
    
    //obtain basic parameters
    int r = Gt.shape(0); // size of time grid
    int N = Gt.shape(1); // size of G matrices
    int m = D.shape(0); // order of diagram
    int P = hyb_F.c.shape(0);

    //initialize diagram
    auto Diagram = nda::array<dcomplex,3>(r,N,N);
    Diagram = 0;

    //iteration over the terms of 2, · · · , m-th hybridization. Note that 1-st hybridization is not decomposed.
    int total_num_diagram = pow(P, m-1);
    for (int num=0;num<total_num_diagram;++num){
        int num0 = num;
        //obtain R2, ... , Rm, store as R[1],...,R[m-1]
        auto R = nda::vector<int>(m);
        for (int v = 1;v<m;++v){
            R[v] = num0 % P;
            num0 = int(num0/P);
        }

        //Phase 1: construct line object L and point object P;
        /* Construct line objects, i.e. functions of (t_s-t_{s-1}) for s =1 , ..., 2m-1
        We store these in L(r,2m,N,N). 
        We initialize them with Green's functions.
        */
        auto L = nda::array<dcomplex,4>(r,2*m,N,N);
        for (int s=1;s<=2*m-1;++s) L(_,s,_,_) = Gt;

        double constant = 1; // the constant term responsible for the current diagram

        /* Construct point objects, i.e. functions at t_s. 
        We store these in P(r,2m,N,N)
        */
        auto P = nda::array<dcomplex,4>(r,2*m,N,N);
        for (int v = 1;v<m;++v){
             
            P(_,D(v,0),_,_) = hyb_F.V_tilde(_,R(v),_,_);
            P(_,D(v,1),_,_) = hyb_F.U_tilde(_,R(v),_,_);
            constant = constant*hyb_F.c(R(v));
            //when w(R(v))>0, we need to modify the line object, and the constant
            if (hyb_F.w0(R(v))>0){
                for (int s = D(v,0)+2; s<D(v,1);++s){
                    for (int k =0;k<r;++k) L(k,s,_,_) = L(k,s,_,_) * k_it(dlr_it(k),hyb_F.w0(R(v)));
                    constant = constant * hyb_F.c(R(v));
                }
            }
        }

        //Phase 2: integrate everything out
        

        
    }
    
    return Diagram;
}