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

#ifndef RAJA_forall_apollo_HPP
#define RAJA_forall_apollo_HPP

#include <omp.h>

#include <Kokkos_Core.hpp>
#include <array>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "RAJA/config.hpp"
#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/internal/fault_tolerance.hpp"
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"
#include "RAJA/policy/apollo/policy.hpp"
#include "RAJA/util/types.hpp"
#include "apollo/Apollo.h"
#include "apollo/Region.h"

// ----------


namespace RAJA
{
namespace policy
{
namespace apollo
{
//
//////////////////////////////////////////////////////////////////////
//
// The following function template switches between various RAJA
// execution policies based on feedback from the Apollo system.
//
//////////////////////////////////////////////////////////////////////
//

#ifndef RAJA_ENABLE_OPENMP
#error \
    "*** RAJA_ENABLE_OPENMP is not defined!" \
    "This build of RAJA requires OpenMP to be enabled! ***"
#endif


static size_t get_code_region_id()
{
  Kokkos::Tools::VariableInfo value;
  static size_t id;
  static bool init;
  if (!init) {
    init = true;
    value.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    value.type = Kokkos::Tools::ValueType::kokkos_value_text;
    value.valueQuantity = kokkos_value_unbounded;

    Kokkos::Tools::SetOrRange range;
    Kokkos::Tools::declareContextVariable("raja.apollo.code_region",
                                          id,
                                          value,
                                          range);
  }
  return id;
}

size_t get_policy_choice_id()
{
  Kokkos::Tools::VariableInfo value;
  static size_t id;
  static bool init;
  if (!init) {
    init = true;
    value.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    value.type = Kokkos::Tools::ValueType::kokkos_value_integer;
    value.valueQuantity = kokkos_value_set;

    Kokkos::Tools::declareTuningVariable("raja.apollo.policy_choice",
                                         id,
                                         value);
  }
  return id;
}

size_t get_loop_length_id()
{
  Kokkos::Tools::VariableInfo value;
  static size_t id;
  static bool init;
  if (!init) {
    init = true;
    value.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    value.type = Kokkos::Tools::ValueType::kokkos_value_floating_point;
    value.valueQuantity = kokkos_value_unbounded;
    Kokkos::Tools::SetOrRange range;

    Kokkos::Tools::declareContextVariable("raja.apollo.loop_length",
                                          id,
                                          value,
                                          range);
  }
  return id;
}

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const apollo_exec &,
                             Iterable &&iter,
                             Func &&loop_body)
{
  static Apollo *apollo = Apollo::instance();
  static Apollo::Region *apolloRegion = nullptr;
  static int policy_index = 0;
  static int num_threads[POLICY_COUNT] = {0};
  static Kokkos::Tools::SetOrRange policy_choices;
  static size_t code_region_variable_id;
  static size_t policy_choice_variable_id;
  static size_t loop_length_variable_id;
  if (apolloRegion == nullptr) {
    Kokkos::initialize();

    code_region_variable_id = get_code_region_id();
    policy_choice_variable_id = get_policy_choice_id();
    loop_length_variable_id = get_loop_length_id();


    // apolloRegion = new Apollo::Region(
    //        1, //num features
    //        code_location.c_str(), // region uid
    //        POLICY_COUNT // num policies
    //);

    // Set the range of thread counts we want to make available for
    // bootstrapping and use by this Apollo::Region.
    num_threads[0] = apollo->ompDefaultNumThreads;
    num_threads[1] = 1;

    num_threads[2] = std::max(2, apollo->numThreadsPerProcCap);
    num_threads[3] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
    num_threads[4] =
        std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
    num_threads[5] =
        std::min(8, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
    num_threads[6] =
        std::min(4, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
    num_threads[7] = 2;
    num_threads[8] = std::max(2, apollo->numThreadsPerProcCap);
    num_threads[9] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
    num_threads[10] =
        std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
    num_threads[11] =
        std::min(8, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
    num_threads[12] =
        std::min(4, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
    num_threads[13] = 2;
    num_threads[14] = std::max(2, apollo->numThreadsPerProcCap);
    num_threads[15] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
    num_threads[16] =
        std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
    num_threads[17] =
        std::min(8, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
    num_threads[18] =
        std::min(4, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
    num_threads[19] = 2;
    policy_choices.set.id = policy_choice_variable_id;
    policy_choices.set.size = POLICY_COUNT;
    policy_choices.set.values = new Kokkos::Tools::VariableValue[POLICY_COUNT];
    for (int64_t x = 0; x < POLICY_COUNT; ++x) {
      policy_choices.set.values[x] =
          Kokkos::Tools::make_variable_value(policy_choice_variable_id, x);
    }
  }

  std::string code_location = apollo->getCallpathOffset();

  // Count the number of elements.
  float num_elements = 0.0;
  num_elements = (float)std::distance(std::begin(iter), std::end(iter));
  std::array<Kokkos::Tools::VariableValue, 2> context_values{
      Kokkos::Tools::make_variable_value(code_region_variable_id,
                                         code_location.c_str()),
      Kokkos::Tools::make_variable_value(loop_length_variable_id, num_elements),
  };
  Kokkos::Tools::VariableValue policy_choice_holder =
      Kokkos::Tools::make_variable_value(policy_choice_variable_id,
                                         policy_choices.set.values[0].value.int_value);

  size_t context = Kokkos::Tools::getNewContextId();
  Kokkos::Tools::declareContextVariableValues(context,2,context_values.data());
  Kokkos::Tools::requestTuningVariableValues(context,1,&policy_choice_holder, &policy_choices);
  //apolloRegion->begin();
  //apolloRegion->setFeature(num_elements);
  
  policy_index = apolloRegion->getPolicyIndex();
  // std::cout << "policy_index " << policy_index << std::endl; //ggout

  switch (policy_index) {
    case 0: {
// std::cout << "OMP defaults" << std::endl; //ggout
#pragma omp parallel
      {
        using RAJA::internal::thread_privatize;
        auto body = thread_privatize(loop_body);  //.get_priv();
        RAJA_EXTRACT_BED_IT(iter);
#pragma omp for
        for (decltype(distance_it) i = 0; i < distance_it; ++i) {
          body.get_priv()(begin_it[i]);
        }
      }
      break;
    }
    case 1: {
      // std::cout << "Sequential" << std::endl;
      using RAJA::internal::thread_privatize;
      auto body = thread_privatize(loop_body);  //.get_priv();
      RAJA_EXTRACT_BED_IT(iter);
      for (decltype(distance_it) i = 0; i < distance_it; ++i) {
        body.get_priv()(begin_it[i]);
      }
      break;
    }
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7: {
// std::cout << "Static num_threads " << num_threads[ policy_index ] <<
// std::endl;
#pragma omp parallel num_threads(num_threads[policy_index])
      {
        using RAJA::internal::thread_privatize;
        auto body = thread_privatize(loop_body);  //.get_priv();
        RAJA_EXTRACT_BED_IT(iter);
#pragma omp for schedule(static)
        for (decltype(distance_it) i = 0; i < distance_it; ++i) {
          body.get_priv()(begin_it[i]);
        }
      }
      break;
    }
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13: {
// std::cout << "Dynamic num_threads " << num_threads[ policy_index ] <<
// std::endl;
#pragma omp parallel num_threads(num_threads[policy_index])
      {
        using RAJA::internal::thread_privatize;
        auto body = thread_privatize(loop_body);  //.get_priv();
        RAJA_EXTRACT_BED_IT(iter);
#pragma omp for schedule(dynamic)
        for (decltype(distance_it) i = 0; i < distance_it; ++i) {
          body.get_priv()(begin_it[i]);
        }
      }
      break;
    }
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19: {
// std::cout << "Guided num_threads " << num_threads[ policy_index ] <<
// std::endl;
#pragma omp parallel num_threads(num_threads[policy_index])
      {
        using RAJA::internal::thread_privatize;
        auto body = thread_privatize(loop_body);  //.get_priv();
        RAJA_EXTRACT_BED_IT(iter);
#pragma omp for schedule(guided)
        for (decltype(distance_it) i = 0; i < distance_it; ++i) {
          body.get_priv()(begin_it[i]);
        }
      }
      break;
    }
  }
  Kokkos::Tools::endContext(context);
  //apolloRegion->end();
}

//////////
}  // namespace apollo
}  // namespace policy
}  // namespace RAJA

#endif  // closing endif for header file include guard
