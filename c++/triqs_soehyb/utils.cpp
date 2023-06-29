#include "utils.hpp"

static constexpr auto _ = nda::range::all;

nda::vector<double> gregwgt(int q) {

  if (q > 15) {
    throw std::runtime_error("q > 15 is not allowed");
  }

  // Get Gregory coefficients

  auto g = nda::vector<double>(16);
  auto gd = nda::vector<double>(16);

  g(0) = -1.0;
  g(1) = 1.0;
  g(2) = -1.0;
  g(3) = 19.0;
  g(4) = -3.0;
  g(5) = 863.0;
  g(6) = -275.0;
  g(7) = 33953.0;
  g(8) = -8183.0;
  g(9) = 3250433.0;
  g(10) = -4671.0;
  g(11) = 13695779093.0;
  g(12) = -2224234463.0;
  g(13) = 132282840127.0;
  g(14) = -2639651053.0;
  g(15) = 111956703448001.0;

  gd(0) = 2.0;
  gd(1) = 12.0;
  gd(2) = 24.0;
  gd(3) = 720.0;
  gd(4) = 160.0;
  gd(5) = 60480.0;
  gd(6) = 24192.0;
  gd(7) = 3628800.0;
  gd(8) = 1036800.0;
  gd(9) = 479001600.0;
  gd(10) = 788480.0;
  gd(11) = 2615348736000.0;
  gd(12) = 475517952000.0;
  gd(13) = 31384184832000.0;
  gd(14) = 689762304000.0;
  gd(15) = 32011868528640000.0;

  for (int i = 0; i < q; ++i) {
    g(i) /= gd(i);
  }

  // Build Pascal matrix

  auto p = nda::matrix<double>(q, q);

  p(0, _) = 1;
  for (int i = 1; i < q; ++i) {
    for (int j = 0; j < q; ++j) {
      p(i, j) = sum(p(i - 1, range(j + 1)));
    }
  }

  // Get Gregory weights

  auto a = nda::matrix<double>(q, q);
  a = 0;
  for (int i = 0; i < q; ++i) {
    for (int j = i; j < q; ++j) {
      a(i, j) = pow(-1, i + j) * p(i, j - i);
    }
  }

  return a * g(range(q));
}

int modwrap(int a, int b) {
  int c = a % b;
  return (c < 0) ? c + b : c;
}

#include <cmath>
#include <iostream>
#include <vector>

std::complex<double> richardson(int m, nda::vector_const_view<dcomplex> u) {
  // Richardson extrapolation on m input values u
  //
  // We imagine u[0] = A(h), u[1] = A(h/2), u[m-1] = A(h/2^(m-1)), and
  // we are trying to compute lim_{h->0} A(h).
  //
  // v contains the 2mth-order Richardson extrapolant,
  // assuming A(h) is described by an even-term Taylor series
  // expansion about h=0.

  auto a = nda::matrix<dcomplex>(m, m);

  a(0, _) = u;

  for (int i = 0; i < m - 1; ++i) {
    for (int j = 0; j < m - i - 1; ++j) {
      a(i + 1, j) =
          (pow(4.0, i + 1) * a(i, j + 1) - a(i, j)) / (pow(4.0, i + 1) - 1.0);
    }
  }

  return a(m - 1, 0);
}