#include "strong_cpl.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/linalg/matmul.hpp>


using namespace cppdlr;
using namespace nda;

// dim is size of hybridization matrix,i.e. impurity size (number of single-particle basis of impurity); 
// N is size of Green's function matrix, i.e. the dimension of impurity Fock space;
// P is number of terms in the decomposition of the hybridization function Delta
// r is the size of the time grid, i.e. the DLR rank

hyb_decomp::hyb_decomp(nda::array_const_view<dcomplex,3> Delta_dlr, nda::vector_const_view<double> dlr_rf, nda::array_const_view<dcomplex,3> Deltat,nda::vector_const_view<double> dlr_it ,double eps){
    // obtain dlr_rank and dim of hyb function
    int dlr_rank = Delta_dlr.shape(0);
    int dim = Delta_dlr.shape(1);

    // prepare for svd
    auto s_vec = nda::array<double, 1>(dim);
    auto U_local = nda::matrix<dcomplex,F_layout>(dim,dim);
    auto VT_local = nda::matrix<dcomplex,F_layout>(dim,dim);
    nda::array<dcomplex, 2, F_layout> a;
    
    //obtain max number of poles 
    int max_num_pole = dim*dlr_rank;

    //prepare places to store w, U, V
    auto U_all = nda::matrix<dcomplex>(dim,max_num_pole);
    auto V_all = nda::matrix<dcomplex>(max_num_pole,dim);
    auto w_all = nda::vector<double>(max_num_pole);

    //loop over all dlr frequencies, do svd, truncate singular value that are too small
    int P = 0;
    for (int i=0;i<dlr_rank; ++i){
        a = Delta_dlr(i,_,_); // I am not happy about making a copy here
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
    U = transpose(U_all(_,range(P))); //transpose U to get the shape that we want
    V = V_all(range(P),_);

    if (real(U(0,0))<0) U = -U; else V = -V; //This minus sign is added because cppdlr kernel k_it(t,w) = -K(t,w).
    
    //calculate the error of the decomposition
    
    int r =Delta_dlr.shape(0);

    std::cout<< "calculating error of the decomposition, this step could be slow since we have not made use of matrice multiplications"<<std::endl;
    auto Deltat_approx =nda::array<dcomplex,3>(r,dim,dim);

    Deltat_approx = 0;
    
    for (int a=0;a<dim;++a){
        for (int b =0 ;b<dim; ++b){
            for (int R = 0;  R<P;++R){
                for (int k=0;k<r;++k) {
                    Deltat_approx(k,a,b) += -k_it(dlr_it(k), w(R))*U(R,a)*V(R,b);
                }
            }
        }
    }
        
    std::cout << "Max error in decomposition of Delta(t): " << max_element(abs((Deltat - Deltat_approx))) << std::endl;
}


hyb_F::hyb_F(hyb_decomp &hyb_decomp, nda::vector_const_view<double> dlr_rf, nda::vector_const_view<double> dlr_it, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag){
    if (F.shape(0)!= hyb_decomp.U.shape(1) ) throw std::runtime_error("F matrices not in the right format, or F matrices and hybridization shape do not match");
    if (F.shape(1)!= F.shape(2)) throw std::runtime_error("F are not square matrices");
    int N = F.shape(1);
    int P = hyb_decomp.V.shape(0);
    int r = dlr_it.shape(0); 
    int vec_len = N*N;

    auto F_reshape = reshape(F,F.shape(0),vec_len);
    auto F_dag_reshape = reshape(F_dag,F_dag.shape(0),vec_len);

    // construct U_c, V_c:
    auto U_c = reshape(matmul(hyb_decomp.U,F_dag_reshape),P,N,N);
    auto V_c = reshape(matmul(hyb_decomp.V,F_reshape),P,N,N);
    
    
    //Finally construct Utilde, Vtilde, and c
    nda::vector<double> c2(P);
    nda::array<dcomplex,4> U2_tilde(r,P,N,N);
    nda::array<dcomplex,4> V2_tilde(r,P,N,N);

    for (int R=0;R<P;++R){
       for (int k=0;k<r;++k){
            U2_tilde(k,R,_,_)= -k_it(dlr_it(k),hyb_decomp.w(R))*U_c(R,_,_);
       }
       if (hyb_decomp.w(R)<0){
            for (int k=0;k<r;++k) V2_tilde(k,R,_,_) = -k_it(dlr_it(k),-hyb_decomp.w(R))*V_c(R,_,_);
            c2(R) = -1/k_it(0,-hyb_decomp.w(R));
       }
       else {
            for (int k=0;k<r;++k) V2_tilde(k,R,_,_) = -k_it(dlr_it(k),hyb_decomp.w(R))*V_c(R,_,_);
            c2(R) = -1/k_it(0,hyb_decomp.w(R)); 
       }
    }
    c = c2;
    U_tilde = U2_tilde;
    V_tilde = V2_tilde;
    w0 = hyb_decomp.w;
}




nda::array<dcomplex,3> Diagram_calc(hyb_F &hyb_F,nda::array_const_view<int,2> D,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Gt,imtime_ops &itops,double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag){
    auto const &dlr_it = itops.get_itnodes();  
    //obtain basic parameters
    int r = Gt.shape(0); // size of time grid
    int N = Gt.shape(1); // size of G matrices
    int m = D.shape(0); // order of diagram
    int P = hyb_F.c.shape(0);
    int dim = F.shape(0);
    

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
        L = 0;
        for (int s=1;s<=2*m-1;++s) L(_,s,_,_) = Gt;
        double constant = 1; // the constant term responsible for the current diagram

        /* Construct point objects, i.e. functions at t_s. Also update line objects.
        This is done by iterating over the 1st,2nd, ..., (m-1)-th interaction lines. 
        (Nothing is done for the vertex connecting to 0. That vertex will be treated differently in the final integration).
        We store these in P(r,2m,N,N).
        */
        auto P = nda::array<dcomplex,4>(r,2*m,N,N);
        P = 0;
        for (int v = 1;v<m;++v){
            constant = constant*hyb_F.c(R(v));
            
            //when w(R(v))>0, we need to modify the line object, and the constant. The point object is assigned to be the identity matrix.
            if (hyb_F.w0(R(v))>0){
                for (int k=0;k<r;++k) P(k,D(v,0),_,_) = eye<dcomplex>(N); 
                for (int k=0;k<r;++k) P(k,D(v,1),_,_) = eye<dcomplex>(N);
                for (int k=0;k<r;++k) L(k,D(v,0),_,_) = matmul(L(k,D(v,0),_,_),hyb_F.V_tilde(k,R(v),_,_));
                for (int k=0;k<r;++k) L(k,D(v,1)-1,_,_) = matmul(hyb_F.U_tilde(k,R(v),_,_), L(k,D(v,1)-1,_,_));
                
                for (int s = D(v,0)+1; s<D(v,1)-1;++s){
                    for (int k =0;k<r;++k) L(k,s,_,_) = L(k,s,_,_) * (-k_it(dlr_it(k),hyb_F.w0(R(v))));
                    constant = constant * hyb_F.c(R(v));
                }
            }
            //when w(R(v))<0, we need only to modify the point object.
            else{
                P(_,D(v,0),_,_) = hyb_F.V_tilde(_,R(v),_,_);
                P(_,D(v,1),_,_) = hyb_F.U_tilde(_,R(v),_,_);
            }
        }
        //Phase 2: integrate everything out
        auto T = nda::array<dcomplex,3>(r,N,N); 
        
        // first, calculate P(t1)*G(t1)
        for (int k = 0;k<r;++k) T(k,_,_) = matmul(P(k,1,_,_),Gt(k,_,_));

        //integrate out indices t1, ..., t(2m-2). In each for loop, first convolution, then multiplication.
        for (int s=1;s<=2*m-2;++s){
            // integrate ts out by convolution:  integral U/Vtilde(t(s+1)-ts) D(ts) dts
            nda::array<dcomplex,3> Lhere = L(_,s,_,_); // Zhen: I am not happy with this copying
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(Lhere),itops.vals2coefs(T));

            //Then multiplication. For vertices that is not connected to zero, this is just a multiplication.
            //calculate U/Vtilde(t(s+1))*T(t(s+1))
            if (s+1 != D(0,1)){
                for (int k = 0;k<r;++k) T(k,_,_) = matmul(P(k,s+1,_,_),T(k,_,_));
            }
            // Do special things for the vertex connecting to 0:
            //T_k = sum_ab Delta_ab(t) Fdag_a *T_k * F_b
            else {
                auto T2 = nda::array<dcomplex,4>(dim,r,N,N);
                T2=0;
                // T2(b,ts) = T(ts)*F_b
                for (int b =0;b<dim;++b){
                    for (int k=0;k<r;++k) T2(b,k,_,_) = matmul(T(k,_,_),F(b,_,_));
                }
                // T2(a,ts) = sum_b Delta(ts)_ab * T2(b,ts)
                for (int k=0;k<r;++k){
                    for (int M=0;M<N;++M){
                        for (int M2 = 0;M2<N;++M2){
                            T2(_,k,M,M2) = matvecmul(Deltat(k,_,_),T2(_,k,M,M2));
                        }
                    }
                }
                // T = sum_a Fdag_a T2(a,ts) 
                T = 0; 
                for (int k=0;k<r;++k){
                    for (int a = 0;a<dim;++a){
                        T(k,_,_) = T(k,_,_)+matmul(F_dag(a,_,_),T2(a,k,_,_));
                    }
                } 
            }
          
        }
       Diagram = Diagram + T*constant;
    }
    
    return Diagram;
}

nda::array<dcomplex,3> OCA_calc(hyb_F &hyb_F,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Gt,imtime_ops &itops,double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag){
    auto const D = nda::array<int,2>{{0,2},{1,3}}; 
    auto const &dlr_it = itops.get_itnodes();  
    //obtain basic parameters
    int r = Gt.shape(0); // size of time grid
    int N = Gt.shape(1); // size of G matrices
    int m = D.shape(0); // order of diagram
    int P = hyb_F.c.shape(0);
    int dim = F.shape(0);
    

    //initialize diagram
    auto Diagram = nda::array<dcomplex,3>(r,N,N);
    Diagram = 0;
    for (int R=0;R<P;++R){
        auto T = nda::array<dcomplex,3>(r,N,N); 
        T = 0;
        // first integrate out t1
        if (hyb_F.w0(R)<0){
            // T(t1) = Vtilde(t1)*G(t1)
            for (int k = 0;k<r;++k) T(k,_,_) = matmul(hyb_F.V_tilde(k,R,_,_),Gt(k,_,_));
            // integrate t1 out: int G(t2-t1)* T(t1) dt1
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(Gt),itops.vals2coefs(T));
        }
        else{
            // T(t2-t1) = G(t2-t1) * Vtilde(t2-t1)
            for (int k = 0;k<r;++k) T(k,_,_) = matmul(Gt(k,_,_),hyb_F.V_tilde(k,R,_,_));
            // integrate t1 out: int T(t2-t1)* G(t1) dt1
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(T),itops.vals2coefs(Gt)); 
        }
        //next, calculate T3(k) = sum_ab Delta_ab(t_k) * Fdag_a * T(k) * F_b
        auto T2 = nda::array<dcomplex,4>(dim,r,N,N);
        T2=0;
        // T2(b,k) = T(k)*F(b)
        for (int b =0;b<dim;++b){
            for (int k=0;k<r;++k) T2(b,k,_,_) = matmul(T(k,_,_),F(b,_,_));
        }
        // T2(a,k) = sum_b Delta_ab(t_k) * T2(b,k)
        for (int k=0;k<r;++k){
            for (int M=0;M<N;++M){
                for (int M2 = 0;M2<N;++M2){
                    T2(_,k,M,M2) = matvecmul(Deltat(k,_,_),T2(_,k,M,M2));
                }
            }
        }
        //T3(k) = sum_a Fdag_a T2(a,k)
        auto T3 =  nda::array<dcomplex,3>(r,N,N);
        T3 = 0; 
        for (int k=0;k<r;++k){
            for (int a = 0;a<dim;++a){
                T3(k,_,_) = T3(k,_,_)+matmul(F_dag(a,_,_),T2(a,k,_,_));
            }
        }
        //finally integrate out t2
        if (hyb_F.w0(R)<0){
            // integrate t2 out: T(t) = int G(t-t2)* T3(t2) dt2
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(Gt),itops.vals2coefs(T3));
            //T(t) = Gt(t) * T(t) 
            for (int k = 0;k<r;++k) T(k,_,_) = matmul(hyb_F.U_tilde(k,R,_,_),T(k,_,_));
        }
        else{
            // T(t-t2) = Utilde(t-t2) * Gt(t-t2)
            for (int k = 0;k<r;++k) T(k,_,_) = matmul(hyb_F.U_tilde(k,R,_,_),Gt(k,_,_));
            // integrate t2 out: T(t) = int T(t-t2)* T3(t2) dt2 
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(T),itops.vals2coefs(T3)); 
        }
        Diagram = Diagram + T*hyb_F.c(R);
    }
    return Diagram;
}