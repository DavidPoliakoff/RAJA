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
 *          This code can utilize values set in the following environment
 *          variables:
 *
 *            - OPENMP_TRACE_OUTPUT_FILE
 *              Where to flush the trace CSV at the program exit.
 *              Defaults to std::cout if nothing is set or path causes
 *              an exception to be thrown.
 *
 *            - OPENMP_TRACE_AS_POLICY
 *              If this trace is used to capture raw OpenMP runtimes for
 *              contrast with RAJA(apollo_exec) policy times, this value
 *              can be used to populate the (int) 'policy_index' column of
 *              the output CSV to identify which RAJA(apollo_exec) policy
 *              it is most closely mimicking.  If it is not set, a value
 *              of -1 will be used.
 *              NOTE: For performance reasons, this value is only read once,
 *                    the first time a RAJA loop is encountered. A loop will
 *                    not pick up any changes made to that environment variable
 *                    after that loop has been evaluated.
 *
 *          The following environment variables are safely read and used
 *          for logging into the trace CSV, but will not change any settings or
 *          impact characteristics of execution:
 *
 *            - HOSTNAME
 *            - OMP_NUM_THREADS
 *            - PMI_RANK / OMPI_COMM_WORLD_RANK
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
#include <sys/time.h>
#include <type_traits>

#include <string>
#include <sstream>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <vector>
#include <tuple>

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

#define NOTE_TIME(__into_this_dbl)                              \
    {                                                           \
        struct timeval t;                                       \
        gettimeofday(&t, NULL);                                 \
        __into_this_dbl = (double)(t.tv_sec + (t.tv_usec/1e6)); \
    }

// ----------

namespace RAJA
{
namespace policy
{
namespace openmp_trace
{

typedef std::vector<
    std::tuple<
        double,
        std::string,
        int,
        std::string,
        int,
        int,
        int,
        double
    >
> TraceVector_t;



inline void replace_all(std::string& input, const std::string& from, const std::string& to) {
	size_t pos = 0;
	while ((pos = input.find(from, pos)) != std::string::npos) {
		input.replace(pos, from.size(), to);
		pos += to.size();
	}
}

inline const char* safe_getenv(const char *var_name, const char *use_this_if_not_found, bool silent=false) {
    char *c = getenv(var_name);
    if (c == NULL) {
        if (not silent) {
            std::cout << "== RAJA(openmp_trace): Looked for " << var_name << " but getenv()" \
                << " did not find anything.  Using '" << use_this_if_not_found \
                << "' (default) instead." << std::endl;
        }
        return use_this_if_not_found;
    } else {
        return c;
    }
}

class OpenMPTraceControl
{
    public:
        ~OpenMPTraceControl() {
            this->end();
        }
        OpenMPTraceControl(const OpenMPTraceControl&) = delete;
        OpenMPTraceControl& operator=(const OpenMPTraceControl&) = delete;

        static OpenMPTraceControl* instance(void) noexcept {
            static OpenMPTraceControl the_instance;
            return &the_instance;
        }

        TraceVector_t *getTraceVectorPtr(void) {
            return &trace_data;
        }

        void end() {
            double flush_time_start = 0.0;
            double flush_time_end   = 0.0;
            NOTE_TIME(flush_time_start);

            std::string flush_filename = safe_getenv("OPENMP_TRACE_OUTPUT_FILE", "stdout");
            if (flush_filename.compare("stdout") == 0) {
                writeTraceVector(std::cout);
            } else {
                try {
                    std::ofstream sink_file(flush_filename, std::fstream::out);
                    writeTraceVector(sink_file);
                } catch (...) {
                    std::cerr << "== RAJA(openmp_trace): ** ERROR ** Could not open the filename specified in" \
                              << " the OPENMP_TRACE_OUTPUT_FILE environment variable:" << std::endl;
                    std::cerr << "== RAJA(openmp_trace): ** ERROR **     \"" << flush_filename << "\"" << std::endl;
                    std::cerr << "== RAJA(openmp_trace): ** ERROR ** Defaulting to std::cout ..." << std::endl;
                    writeTraceVector(std::cout);
                }
            }

            NOTE_TIME(flush_time_end);
            std::cout << "== OPENMP_TRACE: " << std::fixed << (flush_time_end - flush_time_start) \
                      << " seconds to flush trace to CSV." << std::endl;
            return;
        }


    private:
        OpenMPTraceControl() {};
        TraceVector_t trace_data;

        void writeTraceVector(std::ostream &sink) {
            std::cout.precision(17);
            for (auto &t : trace_data) {
                sink \
                    << "TRACE," \
                    << std::get<0>(t) /* exec_time_end */ << "," \
                    << std::get<1>(t) /* node_id */       << "," \
                    << std::get<2>(t) /* comm_rank */     << "," \
                    << std::get<3>(t) /* region_name */   << "," \
                    << std::get<4>(t) /* policy_index */  << "," \
                    << std::get<5>(t) /* num_threads */   << "," \
                    << std::get<6>(t) /* num_elements */  << "," \
                    << std::fixed << std::get<7>(t) /* (exec_time_end - exec_time_begin) */ \
                    << std::endl;
            }
            sink.flush();
        }

};

using openmpTracePolicy = RAJA::omp_parallel_for_exec;

template <typename BODY>
RAJA_INLINE void openmp_trace_executor(BODY body) {
    body(openmpTracePolicy{});
}

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const openmp_trace_exec &, Iterable &&iter, Func &&body)
{
    static bool            initialized_yet   = false;
    static std::string     node_id           = "";
    static int             comm_rank         = -1;
    static int             num_threads       = 0;
    static int             policy_index      = -1;   // <-- need not be used, for apollo tests.
    static std::string     region_name       = "";
    static TraceVector_t  *v                 = nullptr;

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

        node_id        = safe_getenv("HOSTNAME", "default");
        comm_rank      = std::atoi(safe_getenv("PMI_RANK",
                                   safe_getenv("OMPI_COMM_WORLD_RANK",
                                               "-1", true), true));
        num_threads    = std::atoi(safe_getenv("OMP_NUM_THREADS", "-1"));
        policy_index   = std::atoi(safe_getenv("OPENMP_TRACE_AS_POLICY", "-1"));

        v = OpenMPTraceControl::instance()->getTraceVectorPtr();

	    initialized_yet = true;
    }

    // Count the number of elements.
    double num_elements = 0.0;
    num_elements = (double) std::distance(std::begin(iter), std::end(iter));

    double exec_time_begin = 0.0;
    double exec_time_end   = 0.0;

    NOTE_TIME(exec_time_begin);

    openmp_trace_executor([=] (auto pol) mutable {forall_impl(pol, iter, body);});

    NOTE_TIME(exec_time_end);

    v->push_back(
            std::make_tuple(
                exec_time_end,
                node_id,
                comm_rank,
                region_name,
                policy_index,
                num_threads,
                num_elements,
                (exec_time_end - exec_time_begin)
                )
            );

}

//////////
}  // closing brace for apollo namespace
}  // closing brace for policy namespace
}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
