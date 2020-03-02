/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for Apollo-guided execution.
 *
 *          These methods should work on any platform.
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

#ifndef RAJA_forall_openmp_trace_HPP
#define RAJA_forall_openmp_trace_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include <string>
#include <sstream>
#include <functional>
#include <unordered_set>

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"

#include "RAJA/policy/openmp_trace/policy.hpp"
#include "RAJA/internal/fault_tolerance.hpp"

#include "CallpathRuntime.h"




// ----------



namespace RAJA
{
namespace policy
{
namespace openmp_trace
{

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const RAJA::openmp_trace_parallel_for&, Iterable&& iter, Func&& loop_body) {
  RAJA_EXTRACT_BED_IT(iter);
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    loop_body(begin_it[i]);
  }
}


inline void replace_all(std::string& input, const std::string& from, const std::string& to) {
	size_t pos = 0;
	while ((pos = input.find(from, pos)) != std::string::npos) {
		input.replace(pos, from.size(), to);
		pos += to.size();
	}
}

#define NOTE_TIME(__into_this_dbl)                              \
    {                                                           \
        struct timeval t;                                       \
        gettimeofday(&t, NULL);                                 \
        __into_this_dbl = (double)(t.tv_sec + (t.tv_usec/1e6)); \
    }



template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const openmp_trace_parallel_for &, Iterable &&iter, Func &&body)
{
    static bool            initialized_yet   = false;
    static std::string     node_id           = "";
    static int             comm_rank         = -1;
    static int             num_threads       = 0;
    static int             policy_index      = -1;   // <-- need not be used, for apollo tests.
    static std::string     region_name       = "";

    if (not initialized_yet) {
        // Set up this OpenMP wrapper for the first time:       (Runs only once)
        std::stringstream ss_location;
        CallpathRuntime callpath;
        ss_location << callpath.doStackwalk().get(1);
        // Extract out the pointer to our module+offset string and clean it up:
        std::string offsetptr = ss_location.str();
        offsetptr = offsetptr.substr((offsetptr.rfind("/") + 1), (offsetptr.length() - 1));
        replace_all(offsetptr, "(", "_");
        replace_all(offsetptr, ")", "_");
        region_name = offsetptr;

        node_id        = "default" //TODO(chad) : actual answer
        comm_rank      = -1;       //TODO(chad) : actial answer
        num_threads    = std::atoi(getenv("OMP_NUM_THREADS"));             //TODO(chad)
        policy_index   = std::atoi(getenv("TRACE_AS_APOLLO_POLICY_INDEX"); //TODO(chad)

	    initialized_yet = true;
    }

    // Count the number of elements.
    double num_elements = 0.0;
    num_elements = (double) std::distance(std::begin(iter), std::end(iter));


    NOTE_TIME(exec_time_begin);
    body(RAJA::omp_parallel_for_exec{});
    NOTE_TIME(exec_time_end);

    std::cout.precision(17);
    std::cout \
        << "TRACE," \
        << exec_time_end << "," \
        << node_id << ","
        << comm_rank << ","
        << region_name << "," \
        << policy_index << "," \
        << num_threads << "," \
        << num_elements << "," \
        << std::fixed << (exec_time_end - exec_time_begin) \
        << std::endl;

}

//////////
}  // closing brace for apollo namespace
}  // closing brace for policy namespace
}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
