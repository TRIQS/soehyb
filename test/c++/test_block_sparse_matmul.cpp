#include <nda/declarations.hpp>
#include <nda/macros.hpp>
#include <nda/nda.hpp>
#include <block_sparse.hpp>
#include <gtest/gtest.h>

using namespace nda;

TEST(BlockSparseMatMulTest, BDOtimesBO) {
    // test BlockDiagonalOperator * BlockOperator and v v

    nda::array<dcomplex,3> Ablock0{{{0,1},{1,0}},{{1,0},{0,-1}}};
    nda::array<dcomplex,3> Ablock1{{{1,0},{0,-1}},{{0,1},{1,0}}};
    nda::array<dcomplex,3> Ablock2{{{-1}},{{1}}};
    std::vector<nda::array<dcomplex,3>> Ablocks = {Ablock0,Ablock1,Ablock2};
    BlockDiagonalOperator A(Ablocks);

    nda::array<dcomplex,3> Bblock0{{{2},{3}},{{4},{5}}};
    nda::array<dcomplex,3> Bblock1;
    nda::array<dcomplex,3> Bblock2{{{6,7}},{{8,9}}};
    std::vector<nda::array<dcomplex,3>> Bblocks = {Bblock0,Bblock1,Bblock2};
    nda::vector<int> B_block_indices{2,-1,0};
    BlockOperator B(B_block_indices,Bblocks);

    auto AxB = A*B;
    auto BxA = B*A;

    nda::array<dcomplex,3> Cblock0{{{3},{2}},{{4},{-5}}};
    nda::array<dcomplex,3> Cblock2{{{-6,-7}},{{8,9}}};
    std::vector<nda::array<dcomplex,3>> Cblocks = {Cblock0,Bblock1,Cblock2};
    BlockOperator C(B_block_indices,Cblocks);

    nda::array<dcomplex,3> Dblock0{{{-2},{-3}},{{4},{5}}};
    nda::array<dcomplex,3> Dblock2{{{7,6}},{{8,-9}}};
    std::vector<nda::array<dcomplex,3>> Dblocks = {Dblock0,Bblock1,Dblock2};
    BlockOperator D(B_block_indices,Dblocks);

    ASSERT_EQ(AxB.get_block(0),C.get_block(0));
    ASSERT_EQ(AxB.get_block(2),C.get_block(2));
    ASSERT_EQ(BxA.get_block(0),D.get_block(0));
    ASSERT_EQ(BxA.get_block(2),D.get_block(2));
}

TEST(BlockSparseMatMulTest, FtimesBDO) {
    // test BlockDiagonalOperator * FOperator and v v

    nda::array<dcomplex,3> Ablock0{{{0,1},{1,0}},{{1,0},{0,-1}}};
    nda::array<dcomplex,3> Ablock1{{{1,0},{0,-1}},{{0,1},{1,0}}};
    nda::array<dcomplex,3> Ablock2{{{-1}},{{1}}};
    std::vector<nda::array<dcomplex,3>> Ablocks = {Ablock0,Ablock1,Ablock2};
    BlockDiagonalOperator A(Ablocks);

    nda::array<dcomplex,2> Fblock0;
    nda::array<dcomplex,2> Fblock1{{0,1},{1,0}};
    nda::array<dcomplex,2> Fblock2{{0,-1}};
    std::vector<nda::array<dcomplex,2>> Fblocks = {Fblock0,Fblock1,Fblock2};
    nda::vector<int> F_block_indices{-1,0,1};
    FOperator F(F_block_indices,Fblocks);

    auto AxF = A*F;
    auto FxA = F*A;

    nda::array<dcomplex,3> Cblock0;
    nda::array<dcomplex,3> Cblock1{{{0,1},{-1,0}},{{1,0},{0,1}}};
    nda::array<dcomplex,3> Cblock2{{{0,1}},{{0,-1}}};
    std::vector<nda::array<dcomplex,3>> Cblocks = {Cblock0,Cblock1,Cblock2};
    BlockOperator C(F_block_indices,Cblocks);

    nda::array<dcomplex,3> Dblock1{{{1,0},{0,1}},{{0,-1},{1,0}}};
    nda::array<dcomplex,3> Dblock2{{{0,1}},{{-1,0}}};
    std::vector<nda::array<dcomplex,3>> Dblocks = {Cblock0,Dblock1,Dblock2};
    BlockOperator D(F_block_indices,Dblocks);

    ASSERT_EQ(AxF.get_block(1),C.get_block(1));
    ASSERT_EQ(AxF.get_block(2),C.get_block(2));
    ASSERT_EQ(FxA.get_block(1),D.get_block(1));
    ASSERT_EQ(FxA.get_block(2),D.get_block(2));
}