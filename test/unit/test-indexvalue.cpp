//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for Span
///

#include "RAJA/index/IndexValue.hpp"
#include "RAJA_gtest.hpp"

RAJA_INDEX_VALUE(StrongTypeIndex, "Strong Type")

TEST(IndexValue, Construct)
{
  StrongTypeIndex a;
  ASSERT_EQ(0l, *a);
  const StrongTypeIndex b(5);
  ASSERT_EQ(5l, *b);
  ASSERT_EQ(std::string("Strong Type"), StrongTypeIndex::getName());
}

TEST(IndexValue, PrePostIncrement)
{
  StrongTypeIndex a;
  ASSERT_EQ(0l, *a++);
  ASSERT_EQ(1l, *a);
  ASSERT_EQ(2l, *++a);
  ASSERT_EQ(2l, *a);
}

TEST(IndexValue, PrePostDecrement)
{
  StrongTypeIndex a(3);
  ASSERT_EQ(3l, *a--);
  ASSERT_EQ(2l, *a);
  ASSERT_EQ(1l, *--a);
  ASSERT_EQ(1l, *a);
}

TEST(IndexValue, StrongTypesArith)
{
  StrongTypeIndex a(8);
  StrongTypeIndex b(2);

  ASSERT_EQ(StrongTypeIndex(10), a + b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  ASSERT_EQ(StrongTypeIndex(6), a - b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  ASSERT_EQ(StrongTypeIndex(16), a * b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  ASSERT_EQ(StrongTypeIndex(4), a / b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  a += b;
  ASSERT_EQ(StrongTypeIndex(10), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  a -= b;
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  a *= b;
  ASSERT_EQ(StrongTypeIndex(16), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  a /= b;
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);
}

TEST(IndexValue, IndexTypeArith)
{
  StrongTypeIndex a(8);
  RAJA::Index_type b(2);

  ASSERT_EQ(StrongTypeIndex(10), a + b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  ASSERT_EQ(StrongTypeIndex(6), a - b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  ASSERT_EQ(StrongTypeIndex(16), a * b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  ASSERT_EQ(StrongTypeIndex(4), a / b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  a += b;
  ASSERT_EQ(StrongTypeIndex(10), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  a -= b;
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  a *= b;
  ASSERT_EQ(StrongTypeIndex(16), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  a /= b;
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);
}

TEST(IndexValue, StrongTypeCompare)
{
  StrongTypeIndex v1(5);
  StrongTypeIndex v2(6);
  ASSERT_LT(v1, v2);
  ASSERT_LE(v1, v2);
  ASSERT_LE(v1, v1);
  ASSERT_EQ(v1, v1);
  ASSERT_EQ(v2, v2);
  ASSERT_GE(v1, v1);
  ASSERT_GE(v2, v1);
  ASSERT_GT(v2, v1);
  ASSERT_NE(v1, v2);
}

TEST(IndexValue, IndexTypeCompare)
{
  StrongTypeIndex v(5);
  RAJA::Index_type v_lower(4);
  RAJA::Index_type v_higher(6);
  RAJA::Index_type v_same(5);
  ASSERT_LT(v, v_higher);
  ASSERT_LE(v, v_higher);
  ASSERT_LE(v, v_same);
  ASSERT_EQ(v, v_same);
  ASSERT_GE(v, v_same);
  ASSERT_GE(v, v_lower);
  ASSERT_GT(v, v_lower);
  ASSERT_NE(v, v_lower);
  ASSERT_NE(v, v_higher);
}
