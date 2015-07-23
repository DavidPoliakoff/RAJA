/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining RAJA loop execution policies.
 * 
 *          Note: availability of some policies depends on compiler choice.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL 
 *
 ******************************************************************************
 */

#ifndef RAJA_execpolicy_HXX
#define RAJA_execpolicy_HXX


#include "config.hxx"


namespace RAJA {


#if defined(RAJA_COMPILER_ICC)

//
// Segment execution policies
//
struct seq_exec {};
struct simd_exec {};

struct omp_parallel_for_exec {};
//struct omp_for_nowait_exec {};

struct cilk_for_exec {};

//
// Index set segment iteration policies
// 
struct seq_segit {};

struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

struct cilk_for_segit {};


#endif   // end  Intel compilers.....


#if defined(RAJA_COMPILER_GNU) 

//
// Segment execution policies
//
struct seq_exec {};
struct simd_exec {};
struct omp_parallel_for_exec {};
//struct omp_for_nowait_exec {};

//
// Index set segment iteration policies
//
struct seq_segit {};

struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

#endif   // end  GNU compilers.....


#if defined(RAJA_COMPILER_XLC12)

//
// Segment execution policies
//
struct seq_exec {};
struct simd_exec {};
struct omp_parallel_for_exec {};
//struct omp_for_nowait_exec {};

//
// Index set segment iteration policies
//
struct seq_segit {};

struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

#endif   // end  xlc v12 compiler on bgq


#if defined(RAJA_COMPILER_CLANG)

//
// Segment exec policies
//
struct seq_exec {};
struct simd_exec {};
struct omp_parallel_for_exec {};

//
// Index set segment iteration policies
// 
struct seq_segit {};

struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

#endif   // end  CLANG compilers.....


#if defined(RAJA_USE_CUDA)
//
// Segment exec policies
//
struct cuda_exec {};
#endif


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
