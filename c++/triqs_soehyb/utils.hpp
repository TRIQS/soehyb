#pragma once
#include <nda/nda.hpp>

using namespace nda;

/**
 * @brief Contract the last dimension of an array a with the first dimension of
 * an array b
 *
 * @param a  An array or array view of rank at least 2
 * @param b  An array or array view of rank at least 2
 *
 * @return Contraction of the inner dimensions of \p a and \p b
 */
template <nda::MemoryArray Ta, nda::MemoryArray Tb>
nda::array<dcomplex, Ta::rank + Tb::rank - 2, F_layout> arraymult(Ta const &a,
                                                                  Tb const &b) {

  // TODO: Make sure that input arrays are in Fortran layout and that both are
  // dcomplex, or make more generic

  // TODO: Make sure that a and b are both of rank at least 2

  // Get ranks of input arrays
  constexpr int ra = Ta::rank;
  constexpr int rb = Tb::rank;

  // Get inner dimensions of input arrays
  int p = a.shape(ra - 1);
  if (b.shape(0) != p)
    throw std::runtime_error("last dim of a != first dim of b");

  // Get product of outer dimensions of input arrays
  int m = a.size() / p;
  int n = b.size() / p;

  // Reshape input arrays to 2D arrays
  auto a_reshaped = nda::reshape(a, m, p);
  auto b_reshaped = nda::reshape(b, p, n);

  // Get shape of output array
  auto c_shape = std::array<int, ra + rb - 2>();
  for (int i = 0; i < ra - 1; ++i) {
    c_shape[i] = a.shape(i);
  }
  for (int i = ra - 1; i < ra + rb - 2; ++i) {
    c_shape[i] = b.shape(i - ra + 2);
  }

  // Allocate output array
  auto c = nda::array<dcomplex, ra + rb - 2, F_layout>(c_shape);

  // Compute the contraction and return
  reshape(c, m, n) = matmul(a_reshaped, b_reshaped);
  return c;
}

/**
 * @brief Compute Gregory quadrature correction weights
 *
 * @param q  Number of quadrature correction points
 *
 * @return Gregory quadrature correction weights
 *
 * \note Order of accuracy of correction is \p q+1
 * \note Gregory quadrature is unstable for large \p q; use beyond \p q = 8 at
 * your own risk. \p q > 15 is not allowed.
 */
nda::vector<double> gregwgt(int q);

/**
 * @brief Modulus of a number with wrap-around
 *
 * Computes the modulus of \p a with respect to \p b, ensuring that the result
 * is always positive. If modulus of \p a with respect to \p b is negative, \p b
 * is added to the result.
 *
 * @param a  Modulus of \p a is computed
 * @param b  Modulus is computed with respect to \p b
 *
 * @return Modulus of \p a with respect to \p b
 */
int modwrap(int a, int b);

/**
 * @brief Richardson extrapolation for an even-term expansion
 *
 * @param[in] m  Richardson extrapolant of order 2m
 * @param[in] u  \p m input values
 *
 * @return 2mth-order Richardson extrapolant
 *
 * \note  We imagine u[0] = A(h), u[1] = A(h/2), u[m-1] = A(h/2^(m-1)), and
 * we are trying to compute lim_{h->0} A(h). This function returns the
 * 2mth-order Richardson extrapolant, assuming A(h) is described by an even-term
 * Taylor series expansion about h=0.
 */
std::complex<double> richardson(int m, nda::vector_const_view<dcomplex> u);