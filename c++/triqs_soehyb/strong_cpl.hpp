#pragma once
#include "nda/nda.hpp"
#include "cppdlr/cppdlr.hpp"
#include <nda/blas/tools.hpp>

using namespace cppdlr;
using namespace nda;

// dim is size of hybridization matrix,i.e. impurity size (number of single-particle basis of impurity); 
// N is size of Green's function matrix, i.e. the dimension of impurity Fock space;
// P is number of terms in the decomposition of the hybridization function Delta
// r is the size of the time grid, i.e. the DLR rank

/* This is the class for decomposition of a hybridization function. 
    The decomposition is:
        Delta_ab(iv) = sum_R U(R,a)V(R,b)/(iv-w(R))
    Its components are:
        w:    length of P vector: real-valued poles;
        U,V: P*dim array: complex-valued; 
    We construct this decomposition using dlr.  
    Input of constructor:
        Delta_dlr: hyb's dlr coefficients;
        dlr_rf:    dlr real frequencies;
        eps:       
        Deltat: hyb on dlr imagrinary time grid;
        dlr_it: dlr imaginary time grid
    Deltat and dlr_it are only used to check the accuracy of the decomposition.

    Expecting this construction step to be replaced by pole fitting procedures in the future. 

*/
class hyb_decomp {
    public:
    nda::vector<double> w;
    nda::matrix<dcomplex> U;
    nda::matrix<dcomplex> V;
    //should add a constructor that one can choose not to calculate the decomposition error
    //hyb_decomp(nda::array_const_view<dcomplex,3> Delta_dlr, nda::vector_const_view<double> dlr_rf, double eps);
    hyb_decomp(nda::array_const_view<dcomplex,3> Delta_dlr, nda::vector_const_view<double> dlr_rf, nda::array_const_view<dcomplex,3> Deltat,nda::vector_const_view<double> dlr_it, double eps=0);
};

/* This is the class for constructing U_tilde and V_tilde, based on hyb_decomp and the F matrices.
    Its components are:
        U_tilde, V_tilde, both of size (r,P,N,N)
        c, of size (1,P)
        w of size (1,P)
    Input of constructor:
        hyb_decomp: from the class hyb_decomp;
        dlr_rf: dlr real frequencies;
        dlr_it: dlr imaginary times;
        beta: inverse temperature;
        F, F_dag: dim*N*N array;
*/
class hyb_F {
    public:
    nda::vector<double> c;
    nda::vector<double> w;
    nda::array<dcomplex,4> U_tilde;
    nda::array<dcomplex,4> V_tilde;

    hyb_F(hyb_decomp &hyb_decomp, nda::vector_const_view<double> dlr_rf, nda::vector_const_view<double> dlr_it, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag);
};


/* This is the function for evaluating a diagram given its topology D.
    The function input is:
        hyb_F:   info about hyb from the class hyb_F;
        D:       diagram topology, encoded in a m*2 array for m-th order diagram;
        Deltat:  hybridization function on dlr imaginary time grid, r*dim*dim;
        Gt:      Green's function on dlr imaginary time grid,  r*N*N;
        F, F_dag: dim*N*N array;
    The function output is:
        Diagram value on time grid: r*N*N array;
*/
nda::array<dcomplex,3> Diagram_calc(hyb_F &hyb_F,nda::array_const_view<int,2> D,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Gt, imtime_ops &itops,double beta,nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag);


/* This is the function for evaluating the OCA diagram given its topology D.
    The function input is:
        hyb_F:   info about hyb from the class hyb_F;
        Deltat:  hybridization function on dlr imaginary time grid, r*dim*dim;
        Gt:      Green's function on dlr imaginary time grid,  r*N*N;
        F, F_dag: dim*N*N array;
    The function output is:
        Diagram value on time grid: r*N*N array;
*/
nda::array<dcomplex,3> OCA_calc(hyb_F &hyb_F,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Gt,imtime_ops &itops,double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag);
