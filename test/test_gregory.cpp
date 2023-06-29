/**
 * @file test_gregory.cpp
 *
 * @brief Test Gregory weights
 */

#include "../src/utils.hpp"
#include "nda/nda.hpp"
#include <gtest/gtest.h>

using namespace nda;

/**
 * @brief Test Gregory weights for first few orders against known values
 */
TEST(gregory, weights) {

  auto wtrue = nda::vector<double>(4); // Array for true weights

  int q = 1;
  wtrue(0) = -1.0 / 2;
  auto gwgt = gregwgt(q);
  EXPECT_LT(max_element(abs(wtrue(range(q)) - gwgt)), 1e-15);

  q = 2;
  wtrue(0) = -7.0 / 12;
  wtrue(1) = 1.0 / 12;
  gwgt = gregwgt(q);
  EXPECT_LT(max_element(abs(wtrue(range(q)) - gwgt)), 1e-15);

  q = 3;
  wtrue(0) = -5.0 / 8;
  wtrue(1) = 1.0 / 6;
  wtrue(2) = -1.0 / 24;
  gwgt = gregwgt(q);
  EXPECT_LT(max_element(abs(wtrue(range(q)) - gwgt)), 1e-15);

  q = 4;
  wtrue(0) = -469.0 / 720;
  wtrue(1) = 59.0 / 240;
  wtrue(2) = -29.0 / 240;
  wtrue(3) = 19.0 / 720;
  gwgt = gregwgt(q);
  EXPECT_LT(max_element(abs(wtrue(range(q)) - gwgt)), 1e-15);
}