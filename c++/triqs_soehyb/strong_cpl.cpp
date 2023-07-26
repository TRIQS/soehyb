#include "strong_cpl.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <nda/linalg/matmul.hpp>


using namespace cppdlr;
using namespace nda;

// n is size of hybridization matrix,i.e. impurity size (number of single-particle basis of impurity); 
// N is size of Green's function matrix, i.e. the dimension of impurity Fock space;
// P is number of terms in the decomposition of the hybridization function Delta
// r is the size of the time grid, i.e. the DLR rank

hyb_decomp::hyb_decomp(nda::array_const_view<dcomplex,3> Matrices, nda::vector_const_view<double> poles,double eps){
    // obtain dlr_rank and n of hyb function
    int p = Matrices.shape(0);
    int n = Matrices.shape(1);

    // prepare for svd
    auto s_vec = nda::array<double, 1>(n);
    auto U_local = nda::matrix<dcomplex,F_layout>(n,n);
    auto VT_local = nda::matrix<dcomplex,F_layout>(n,n);
    auto a =  nda::array<dcomplex, 2, F_layout>(n,n);
    
    //obtain max number of poles 
    int max_num_pole = n*p;

    //prepare places to store w, U, V
    auto U_all = nda::matrix<dcomplex>(n,max_num_pole);
    auto V_all = nda::matrix<dcomplex>(max_num_pole,n);
    auto w_all = nda::vector<double>(max_num_pole);

    //loop over all dlr frequencies, do svd, truncate singular value that are too small
    int P = 0;
    for (int i=0;i<p; ++i){
        a = Matrices(i,_,_); // I am not happy about making a copy here
        nda::lapack::gesvd(a,s_vec,U_local,VT_local);
        for (int d=0; d<n; ++d){
            if (s_vec(d)>eps){
                w_all(P) = poles(i);
                U_all(_,P) = U_local(_,d)*sqrt(s_vec(d));
                V_all(P,_) = VT_local(d,_)*sqrt(s_vec(d));
                P +=1;
            }
        }
    }
    w = w_all(range(P));
    U = transpose(U_all(_,range(P))); //transpose U to get the shape that we want
    V = V_all(range(P),_);

    //if (real(U(0,0))<0) U = -U; else V = -V; //This minus sign is added because cppdlr kernel k_it(t,w) = -K(t,w).

}
void hyb_decomp::check_accuracy(nda::array_const_view<dcomplex,3> Deltat,nda::vector_const_view<double> dlr_it){
    int r =Deltat.shape(0);
    int n = Deltat.shape(1);
    int P = U.shape(0);
    
    std::cout<< "calculating error of the decomposition of hybridization"<<std::endl;
    auto Deltat_approx =nda::array<dcomplex,3>(r,n,n);
    Deltat_approx = 0;
    
    for (int k=0;k<r;++k){
        for (int R = 0;  R<P;++R){
            for (int a=0;a<n;++a) {
                for (int b =0 ;b<n; ++b){
                    Deltat_approx(k,a,b) += k_it(dlr_it(k), w(R))*U(R,a)*V(R,b);
                }
            }
        }
    }
    // std::cout<<Deltat<<std::endl<<Deltat_approx; 
    std::cout << "Max error in decomposition of Delta(t): " << max_element(abs((Deltat - Deltat_approx))) << std::endl; 
}

hyb_F::hyb_F(hyb_decomp &hyb_decomp, nda::vector_const_view<double> dlr_rf, nda::vector_const_view<double> dlr_it, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag){
    if (F.shape(0)!= hyb_decomp.U.shape(1) ) throw std::runtime_error("F matrices not in the right format, or F matrices and hybridization shape do not match");
    if (F.shape(1)!= F.shape(2)) throw std::runtime_error("F are not square matrices");
    int N = F.shape(1);
    int P = hyb_decomp.V.shape(0);
    int r = dlr_it.shape(0); 
    int vec_len = N*N;

    auto U_c = arraymult(hyb_decomp.U,F_dag);
    auto V_c = arraymult(hyb_decomp.V,F); 
    
    
    //Finally construct Utilde, Vtilde, and c
    c = nda::vector<double>(P);
    U_tilde = nda::array<dcomplex,4>(P,r,N,N);
    U_tilde = 0;
    V_tilde = nda::array<dcomplex,4>(P,r,N,N);
    V_tilde = 0;
    K_matrix = nda::array<double,2>(P,r);
    for (int R=0;R<P;++R){
       for (int k=0;k<r;++k) K_matrix(R,k) = k_it(dlr_it(k),hyb_decomp.w(R)); 
    }

    for (int R=0;R<P;++R){
       for (int k=0;k<r;++k){
            U_tilde(R,k,_,_)= K_matrix(R,k)*U_c(R,_,_);
       }
       if (hyb_decomp.w(R)<0){
            for (int k=0;k<r;++k) V_tilde(R,k,_,_) = k_it(dlr_it(k),-hyb_decomp.w(R))*V_c(R,_,_);
            c(R) = 1/k_it(0,-hyb_decomp.w(R));
       }
       else {
            for (int k=0;k<r;++k) V_tilde(R,k,_,_) = K_matrix(R,k)*V_c(R,_,_);
            c(R) = 1/k_it(0,hyb_decomp.w(R)); 
       }
    }
    w = hyb_decomp.w;
}

nda::array<dcomplex,3> G_Diagram_calc(hyb_F &hyb_F_self,hyb_F &hyb_F_reflect,nda::array_const_view<int,2> D,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Deltat_reflect,nda::array_const_view<dcomplex,3> Gt, imtime_ops &itops,double beta,nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag,nda::vector_const_view<int> fb, bool backward){
    auto const &dlr_it = itops.get_itnodes();  
    //obtain basic parameters
    int r = Gt.shape(0); // size of time grid
    int N = Gt.shape(1); // size of G matrices
    int m = D.shape(0); // order of diagram
    int P = hyb_F_self.c.shape(0);
    int n = F.shape(0);
    

    //initialize diagram
    auto Diagram = nda::array<dcomplex,3>(r,n,n);
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
        //This is done exactly the same as in Sigma diagrams
        auto line = nda::array<dcomplex,4>(2*m,r,N,N);
        line = 0;
        for (int s=1;s<=2*m-1;++s) line(s,_,_,_) = Gt;
        double constant = 1; 
        auto vertex = nda::array<dcomplex,4>(2*m,r,N,N);
        vertex = 0;

        
        for (int v = 1;v<m;++v) {
            if (fb(v)==0)  cut_hybridization(v,R(v), D, constant, hyb_F_self.U_tilde(R(v),_,_,_),hyb_F_self.V_tilde(R(v),_,_,_), line, vertex,hyb_F_self.c(R(v)),hyb_F_self.w(R(v)),hyb_F_self.K_matrix(R(v),_) ,r, N);
            else cut_hybridization(v,R(v), D, constant, hyb_F_reflect.U_tilde(R(v),_,_,_),hyb_F_reflect.V_tilde(R(v),_,_,_), line, vertex,hyb_F_reflect.c(R(v)),hyb_F_reflect.w(R(v)),hyb_F_reflect.K_matrix(R(v),_) ,r, N); 
        } 
        //TODO: the integrating part of G diagrams. Have to figure out what is happening here.
        
        //Phase 2: integrate out the stuff on the right
        auto T = nda::array<dcomplex,3>(r,N,N); 
        
        // first, calculate P(t1)*G(t1)
        for (int k = 0;k<r;++k) T(k,_,_) = matmul(vertex(1,k,_,_),Gt(k,_,_));

        //integrate out the stuff on the right. In each for loop, first convolution, then multiplication.
        for (int s=1;s<D(0,1);++s){
            // integrate ts out by convolution:  integral L_s(t(s+1)-ts) D(ts) dts
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(line(s,_,_,_)),itops.vals2coefs(T));
            //Then multiplication. For vertices that is not connected to zero, this is just a multiplication.
            //Do special things for the vertex connecting to 0.
            if (s+1 != D(0,1)) multiplicate_onto(vertex(s+1,_,_,_),T);
        }

        //Phase 2.5: integrate out the stuff on the right
        //First, change the vertex objects from v(t)->v(beta-t)
        for (int s=2*m-1;s>D(0,1);--s) vertex(s,_,_,_) = itops.reflect(vertex(s,_,_,_));
        auto T_left = nda::array<dcomplex,3>(r,N,N); 
        
        // first, calculate P(t1)*G(t1)
        for (int k = 0;k<r;++k) T_left(k,_,_) = matmul(Gt(k,_,_),vertex(2*m-1,k,_,_));

        //integrate out the stuff on the right. In each for loop, first convolution, then multiplication.
        for (int s=2*m-2;s>D(0,1);--s){
            // integrate ts out by convolution:  integral L_s(t(s+1)-ts) D(ts) dts
            T_left = itops.tconvolve(beta, Fermion,itops.vals2coefs(T_left),itops.vals2coefs(line(s,_,_,_)));
            //Then multiplication. For vertices that is not connected to zero, this is just a multiplication.
            //Do special things for the vertex connecting to 0.
            if (s-1 != D(0,1)) multiplicate_onto_left(T_left,vertex(s-1,_,_,_));
        }
        //revert back from (beta-t) to t
        T_left = itops.reflect(T_left); 

        auto GF_dag = nda::array<dcomplex,4>(n,r,N,N);
        auto GF = nda::array<dcomplex,4>(n,r,N,N);
        auto GF_left_dag = nda::array<dcomplex,4>(n,r,N,N);
        auto GF_left = nda::array<dcomplex,4>(n,r,N,N);
        for (int b=0;b<n;++b){
            for (int k = 0;k<r;++k) GF_dag(b,k,_,_) = matmul(T(k,_,_), F_dag(b,_,_));
            for (int k = 0;k<r;++k) GF(b,k,_,_) = matmul(T(k,_,_), F(b,_,_));
            for (int k = 0;k<r;++k) GF_left_dag(b,k,_,_) = matmul(T_left(k,_,_), F_dag(b,_,_));
            for (int k = 0;k<r;++k) GF_left(b,k,_,_) = matmul(T_left(k,_,_), F(b,_,_));  
        }
        for (int b=0;b<n;++b){
            for (int a=0;a<n;++a){
                for (int k = 0;k<r;++k){
                    Diagram(k,a,b) += trace(matmul(GF_left(a,k,_,_),GF_dag(b,k,_,_)));
                    Diagram(k,a,b) += trace(matmul(GF_left_dag(a,k,_,_),GF(b,k,_,_))); 
                }
            }
        }
    }   
    return Diagram; 
}



nda::array<dcomplex,3> Sigma_Diagram_calc(hyb_F &hyb_F_self,hyb_F &hyb_F_reflect,nda::array_const_view<int,2> D,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Deltat_reflect,nda::array_const_view<dcomplex,3> Gt,imtime_ops &itops,double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag, nda::vector_const_view<int> fb, bool backward){
    auto const &dlr_it = itops.get_itnodes();  
    //obtain basic parameters
    int r = Gt.shape(0); // size of time grid
    int N = Gt.shape(1); // size of G matrices
    int m = D.shape(0); // order of diagram
    int P = hyb_F_self.c.shape(0);
    int n = F.shape(0);
    
    

    //initialize diagram
    auto Diagram = nda::array<dcomplex,3>(r,N,N);
    Diagram = 0;

    if (m==1){
        Diagram = Gt;
        special_summation(Diagram, F, F_dag, Deltat,Deltat_reflect, n, r, N, backward); 
        return Diagram;
    }

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
        auto line = nda::array<dcomplex,4>(2*m,r,N,N);
        line = 0;
        for (int s=1;s<=2*m-1;++s) line(s,_,_,_) = Gt;
        double constant = 1; // the constant term responsible for the current diagram
        //construct vertex object
        auto vertex = nda::array<dcomplex,4>(2*m,r,N,N);
        vertex = 0;

        //Cutting 1,2, ..., (m-1)-th hybridization lines.
        //for (int v = 1;v<m;++v) cut_hybridization(v,R(v), D, constant, hyb_F_self,hyb_F_reflect, line, vertex, r, N);
        for (int v = 1;v<m;++v) {
            if (fb(v)==0)  cut_hybridization(v,R(v), D, constant, hyb_F_self.U_tilde(R(v),_,_,_),hyb_F_self.V_tilde(R(v),_,_,_), line, vertex,hyb_F_self.c(R(v)),hyb_F_self.w(R(v)),hyb_F_self.K_matrix(R(v),_) ,r, N);
            else cut_hybridization(v,R(v), D, constant, hyb_F_reflect.U_tilde(R(v),_,_,_),hyb_F_reflect.V_tilde(R(v),_,_,_), line, vertex,hyb_F_reflect.c(R(v)),hyb_F_reflect.w(R(v)),hyb_F_reflect.K_matrix(R(v),_) ,r, N); 
        }
        //Phase 2: integrate everything out
        auto T = nda::array<dcomplex,3>(r,N,N); 
        
        // first, calculate P(t1)*G(t1)
        for (int k = 0;k<r;++k) T(k,_,_) = matmul(vertex(1,k,_,_),Gt(k,_,_));

        //integrate out indices t1, ..., t(2m-2). In each for loop, first convolution, then multiplication.
        for (int s=1;s<=2*m-2;++s){
            // integrate ts out by convolution:  integral L_s(t(s+1)-ts) D(ts) dts
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(line(s,_,_,_)),itops.vals2coefs(T));
            //Then multiplication. For vertices that is not connected to zero, this is just a multiplication.
            //Do special things for the vertex connecting to 0.
            if (s+1 != D(0,1)) multiplicate_onto(vertex(s+1,_,_,_),T);
            else special_summation(T, F, F_dag, Deltat,Deltat_reflect, n, r, N, backward);
        }
       Diagram = Diagram + T*constant;
    }
    
    return Diagram;
}

nda::array<dcomplex,3> Sigma_Diagram_calc_sum_all(hyb_F &hyb_F_self,hyb_F &hyb_F_reflect,nda::array_const_view<int,2> D,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Deltat_reflect,nda::array_const_view<dcomplex,3> Gt,imtime_ops &itops,double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag){
    int r = Gt.shape(0); // size of time grid
    int N = Gt.shape(1); // size of G matrices
    int m = D.shape(0); // order of diagram
    auto Diagram = nda::array<dcomplex,3>(r,N,N);
    Diagram = 0;
    int total_num_fb_diagram = pow(2, m-1);
    for (int num=0;num<total_num_fb_diagram;++num){
        int num0 = num;
        auto fb = nda::vector<int>(m);
        for (int v = 1;v<m;++v){
            fb[v] = num0 % 2;
            num0 = int(num0/2);
        }
       // std::cout<<fb<<std::endl;
        Diagram += Sigma_Diagram_calc(hyb_F_self,hyb_F_reflect,D,Deltat,Deltat_reflect, Gt,itops,beta, F,  F_dag,  fb, true);
    }
    return Diagram;
}

nda::array<dcomplex,3> Sigma_OCA_calc(hyb_F &hyb_F,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Deltat_reflect,nda::array_const_view<dcomplex,3> Gt,imtime_ops &itops,double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag, bool backward){
    auto const D = nda::array<int,2>{{0,2},{1,3}}; 
    auto const &dlr_it = itops.get_itnodes();  
    //obtain basic parameters
    int r = Gt.shape(0); // size of time grid
    int N = Gt.shape(1); // size of G matrices
    int m = D.shape(0); // order of diagram
    int P = hyb_F.c.shape(0);
    int n = F.shape(0);

    
    //initialize diagram
    auto Diagram = nda::array<dcomplex,3>(r,N,N);
    Diagram = 0;
    for (int R=0;R<P;++R){

        auto T = nda::array<dcomplex,3>(r,N,N); 
        T = 0;
        // first integrate out t1
        if (hyb_F.w(R)<0){
            // T(t1) = Vtilde(t1)*G(t1)
            for (int k = 0;k<r;++k) T(k,_,_) = matmul(hyb_F.V_tilde(R,k,_,_),Gt(k,_,_));
            // integrate t1 out: int G(t2-t1)* T(t1) dt1
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(Gt),itops.vals2coefs(T));
        }
        else{
            // T(t2-t1) = G(t2-t1) * Vtilde(t2-t1)
            for (int k = 0;k<r;++k) T(k,_,_) = matmul(Gt(k,_,_),hyb_F.V_tilde(R,k,_,_));
            // integrate t1 out: int T(t2-t1)* G(t1) dt1
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(T),itops.vals2coefs(Gt)); 
        }
        //next, calculate T3(k) = sum_ab Delta_ab(t_k) * Fdag_a * T(k) * F_b
        special_summation(T, F, F_dag, Deltat,Deltat_reflect, n, r, N,backward); 
        
        //finally integrate out t2
        if (hyb_F.w(R)<0){
            // integrate t2 out: T(t) = int G(t-t2)* T3(t2) dt2
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(Gt),itops.vals2coefs(T));
            
            //T(t) = Utilde(t) * T(t) 
            
            for (int k = 0;k<r;++k) T(k,_,_) = matmul(hyb_F.U_tilde(R,k,_,_),T(k,_,_));
           // std::cout<<T<<std::endl;
        }
        else{
            // T(t-t2) = Utilde(t-t2) * Gt(t-t2)
            auto T3 = nda::array<dcomplex,3>(r,N,N);  
            for (int k = 0;k<r;++k) T3(k,_,_) = matmul(hyb_F.U_tilde(R,k,_,_),Gt(k,_,_));
            // integrate t2 out: T(t) = int T(t-t2)* T3(t2) dt2 
            T = itops.tconvolve(beta, Fermion,itops.vals2coefs(T3),itops.vals2coefs(T)); 
        }
        Diagram = Diagram + T*hyb_F.c(R);
        
    }
    return Diagram;
}





//void cut_hybridization(int v,int &Rv,nda::array_const_view<int,2> D, double &constant,  hyb_F &hyb_F_self,  hyb_F &hyb_F_reflect, nda::array_view<dcomplex,4> line, nda::array_view<dcomplex,4> vertex,int &r, int &N){
void cut_hybridization(int v,int &Rv,nda::array_const_view<int,2> D, double &constant,  nda::array_const_view<dcomplex, 3>U_tilde_here,  nda::array_const_view<dcomplex, 3>V_tilde_here, nda::array_view<dcomplex,4> line, nda::array_view<dcomplex,4> vertex, double & chere, double & w_here,nda::array_const_view<double,1> K_matrix_here, int &r, int &N){
   constant *= chere;
            //when w(R(v))>0, we need to modify the line object, and the constant. The point object is assigned to be the identity matrix.
            if (w_here>0){
                for (int k=0;k<r;++k){
                    vertex(D(v,0),k,_,_) = eye<dcomplex>(N); 
                    vertex(D(v,1),k,_,_) = eye<dcomplex>(N);
                    line(D(v,0),k,_,_) = matmul(line(D(v,0),k,_,_),V_tilde_here(k,_,_));
                    line(D(v,1)-1,k,_,_) = matmul(U_tilde_here(k,_,_), line(D(v,1)-1,k,_,_));
                } 
                for (int s = D(v,0)+1; s<D(v,1)-1;++s){
                    for (int k =0;k<r;++k) line(s,k,_,_) *=K_matrix_here(k);
                    constant *= chere;
                }
            }
            //when w(R(v))<0, we need only to modify the point object.
            else{
                vertex(D(v,0),_,_,_) =V_tilde_here;
                vertex(D(v,1),_,_,_) = U_tilde_here;
            } 
}
void special_summation(nda::array_view<dcomplex,3> T, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Deltat_reflect, int &n, int &r, int &N, bool backward){
    auto T2 = nda::array<dcomplex,4>(n,r,N,N);
    T2=0;
    // T2(b,ts) = T(ts)*F_b
    for (int b =0;b<n;++b){
        for (int k=0;k<r;++k) T2(b,k,_,_) = matmul(T(k,_,_),F(b,_,_));
    }
    // T2(a,ts) = sum_b Delta(ts)_ab * T2(b,ts)
    for (int k=0;k<r;++k){
        for (int M=0;M<N;++M){
            for (int M2 = 0;M2<N;++M2) T2(_,k,M,M2) = matvecmul(Deltat(k,_,_),T2(_,k,M,M2));
        }
    }
    // T = sum_a Fdag_a T2(a,ts) 
    auto T3 = nda::array<dcomplex,3>(T.shape());
    T3=0;
    for (int k=0;k<r;++k){
        for (int a = 0;a<n;++a) T3(k,_,_) = T3(k,_,_)+matmul(F_dag(a,_,_),T2(a,k,_,_));
    }  
    if (backward==true){
        T2=0;
        for (int b =0;b<n;++b){
           for (int k=0;k<r;++k) T2(b,k,_,_) = matmul(T(k,_,_),F_dag(b,_,_));
        }
        for (int k=0;k<r;++k){
            for (int M=0;M<N;++M){
                for (int M2 = 0;M2<N;++M2) T2(_,k,M,M2) = matvecmul(Deltat_reflect(k,_,_),T2(_,k,M,M2));
            }
        }
        // T = sum_a Fdag_a T2(a,ts) 
        for (int k=0;k<r;++k){
            for (int a = 0;a<n;++a) T3(k,_,_) = T3(k,_,_)+matmul(F(a,_,_),T2(a,k,_,_));
        }  
    }
    T = T3;
}

void multiplicate_onto(nda::array_const_view<dcomplex,3> Ft, nda::array_view<dcomplex,3> Gt){
    int r = Gt.shape(0);
    for (int k = 0;k<r;++k) Gt(k,_,_) = matmul(Ft(k,_,_),Gt(k,_,_));
}
void multiplicate_onto_left(nda::array_view<dcomplex,3> Ft, nda::array_const_view<dcomplex,3> Gt){
    int r = Gt.shape(0);
    for (int k = 0;k<r;++k) Ft(k,_,_) = matmul(Ft(k,_,_),Gt(k,_,_));
}