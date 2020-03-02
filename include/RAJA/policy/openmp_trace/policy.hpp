/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA openmp_trace policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_trace_HPP
#define RAJA_policy_openmp_trace_HPP

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace policy
{
namespace openmp_trace
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

struct openmp_trace_parallel_for : make_policy_pattern_launch_platform_t<Policy::openmp_trace,
                                                        Pattern::forall,
                                                        Launch::undefined,
                                                        Platform::host> {
};


///
/// Segment execution policies
///

struct openmp_trace_region : make_policy_pattern_launch_platform_t<Policy::openmp_trace,
                                                          Pattern::region,
                                                          Launch::sync,
                                                          Platform::host> {
};

struct openmp_trace_exec : make_policy_pattern_launch_platform_t<Policy::openmp_trace,
                                                        Pattern::forall,
                                                        Launch::undefined,
                                                        Platform::host> {
};

///
/// Index set segment iteration policies
///
using openmp_trace_segit = openmp_trace_exec;

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct openmp_trace_reduce : make_policy_pattern_launch_platform_t<Policy::openmp_trace,
                                                          Pattern::forall,
                                                          Launch::undefined,
                                                          Platform::host> {
};

}  // end namespace openmp_trace
}  // end namespace policy

using policy::openmp_trace::openmp_trace_exec;
using policy::openmp_trace::openmp_trace_region;
using policy::openmp_trace::openmp_trace_segit;
using policy::openmp_trace::openmp_trace_reduce;

using policy::openmp_trace::openmp_trace_parallel_for;

///
///////////////////////////////////////////////////////////////////////
///
/// Shared memory policies
///
///////////////////////////////////////////////////////////////////////
///

struct openmp_trace_shmem {
};

}  // closing brace for RAJA namespace

#endif
