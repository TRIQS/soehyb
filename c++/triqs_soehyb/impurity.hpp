#pragma once
#include "nda/nda.hpp"
#include "cppdlr/cppdlr.hpp"
#include <nda/blas/tools.hpp>
#include "strong_cpl.hpp"

using namespace cppdlr;
using namespace nda;

/**
@note n is size of hybridization matrix,i.e. impurity size (number of single-particle basis of impurity); 
@note N is size of Green's function matrix, i.e. the dimension of impurity Fock space;
@note P is number of terms in the decomposition of the hybridization function Delta
@note r is the size of the time grid, i.e. the DLR rank
* */

/**
  * @class fastdiagram
  * @brief Class responsible for fast diagram calculation of a given impurity problem using hybridization expansion.
* */
class fastdiagram{
    public:
    /** 
    * @brief Constructor for fastdiagram
    * @param[in] beta inverse temperature
    * @param[in] lambda DLR cutoff parameter
    * @param[in] eps DLR accuracy tolerance
    * @param[in] Deltat hybridization function in imaginary time, nda array of size r*n*n
    * @param[in] F impurity annihilation operator in pseudo-particle space, of size n*N*N
    * @param[in] F_dag impurity creation operator in pseudo-particle space, of size n*N*N
    * @param[in] poledlrflag flag for whether to use dlr for pole expansion. True for using dlr. False has not been implemented yet. 
    * */
    fastdiagram(double beta, double lambda, double eps, nda::array<dcomplex,3> Deltat,nda::array<dcomplex,3> F, nda::array<dcomplex,3> F_dag, bool poledlrflag);

    /** 
    * @brief Compute pseudo-particle self energy diagram of certain order, given pseudo-particle Green's function G(t)
    * @param[in] Gt pseudo-particle Green's function G(t), of size r*N*N
    * @param[in] order diagram order: "NCA", "OCA" or "TCA"
    * @return pseudo-particle self energy diagram, r*N*N
    * */ 
    nda::array<dcomplex,3> Sigma_calc(nda::array_const_view<dcomplex,3> Gt, std::string order);

    /** 
    * @brief Compute impurity Green's function diagram of certain order, given pseudo-particle Green's function G(t)
    * @param[in] Gt pseudo-particle Green's function G(t), of size r*N*N
    * @param[in] order diagram order: "NCA", "OCA" or "TCA"
    * @return impurity Green's function diagram, r*n*n
    * */ 
    nda::array<dcomplex,3> G_calc(nda::array_const_view<dcomplex,3> Gt, std::string order); 

    private:
    double beta; //inverse temperature
    double lambda; // DLR cutoff parameter
    nda::vector<double> dlr_rf; // DLR real frequencies
    imtime_ops itops; // DLR imaginary time objects from cppdlr
    nda::vector<double> dlr_it; // DLR imaginary time nodes
    
    nda::array<dcomplex,3> F; // impurity annihilation operator in pseudo-particle space, of size n*N*N
    nda::array<dcomplex,3> F_dag; // impurity creation operator in pseudo-particle space, of size n*N*N

    nda::array<dcomplex,3> Deltat; //hybridization function in imaginary time, nda array of size r*n*n
    nda::array<dcomplex,3> Deltat_reflect; // Delta(beta-t), of size r*n*n

    hyb_F Delta_F; // Compression of Delta(t) and F, F_dag matrices
    hyb_F Delta_F_reflect; // Compression of Delta(-t) and F, F_dag matrices

    nda::array<int,2> D_NCA; // NCA diagram information
    nda::array<int,2> D_OCA; // OCA diagram information
    nda::array<int,2> D_TCA_1; //TCA 1st diagram information
    nda::array<int,2> D_TCA_2; //TCA 2nd diagram information
    nda::array<int,2> D_TCA_3; //TCA 3rd diagram information
    nda::array<int,2> D_TCA_4; //TCA 4th diagram information
};


