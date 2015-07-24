/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template
 *          methods for SIMD execution.
 *
 *          These methods should work on any platform.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_simd_HXX
#define RAJA_forall_simd_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

#include "execpolicy.hxx"

#include "fault_tolerance.hxx"

namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range index sets.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const Index_type begin, const Index_type end, 
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range, including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(simd_exec,
                   const Index_type begin, const Index_type end,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type loop_end = end - begin + 1;

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range set object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range set object, including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(simd_exec,
                   const RangeSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type loop_end = iseg.getEnd() - iseg.getBegin() + 1;

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD minloc reduction over index range.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const Index_type begin, const Index_type end,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD minloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const RangeSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD maxloc reduction over index range.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const Index_type begin, const Index_type end,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD maxloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const RangeSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD sum reduction over index range.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const Index_type begin, const Index_type end,
                T* sum,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD sum reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const RangeSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range index sets with stride.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range with stride.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const Index_type begin, const Index_type end, 
            const Index_type stride,
            LOOP_BODY loop_body)
{  
   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range with stride, including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(simd_exec,
                   const Index_type begin, const Index_type end,
                   const Index_type stride,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type loop_end = (end-begin)/stride + 1;

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const RangeStrideSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over range index set with stride object,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(simd_exec,
                   const RangeStrideSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type begin    = iseg.getBegin();
   const Index_type stride   = iseg.getStride();
   const Index_type loop_end = iseg.getLength();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD minloc reduction over index range with stride.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const Index_type begin, const Index_type end,
                   const Index_type stride,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD minloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const RangeStrideSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD maxloc reduction over index range with stride.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const Index_type begin, const Index_type end,
                   const Index_type stride,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD maxloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const RangeStrideSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD sum reduction over index range with stride.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const Index_type begin, const Index_type end,
                const Index_type stride,
                T* sum,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD sum reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const RangeStrideSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segments.
//
// NOTE: These operations will not vectorize, so we force sequential
//       execution.  Hence, they are "fake" SIMD operations.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const Index_type* __restrict__ idx, const Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over indices in indirection array,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(simd_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}


/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const ListSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over list segment object,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(simd_exec,
                   const ListSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD minloc reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD minloc reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const ListSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD maxloc reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD maxloc reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const ListSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD sum reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const Index_type* __restrict__ idx, const Index_type len,
                T* sum,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], sum );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD sum reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const ListSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// SIMD execution policy does not apply to iteration over index 
// set segments, only to execution of individual segments.
//
//////////////////////////////////////////////////////////////////////
//


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
