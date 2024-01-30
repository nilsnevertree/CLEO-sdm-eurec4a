/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: kokkosaliases_sd.hpp
 * Project: superdrops
 * Created Date: Saturday 14th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Wednesday 25th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * aliases for Kokkos superdrop views
 */

#ifndef LIBS_SUPERDROPS_KOKKOSALIASES_SD_HPP_
#define LIBS_SUPERDROPS_KOKKOSALIASES_SD_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Random.hpp>

#include "./superdrop.hpp"

/* Execution Spaces and Memory for Parallelism */
using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostSpace = Kokkos::DefaultHostExecutionSpace;

/* Superdrop views and subviews */
using viewd_supers = Kokkos::View<Superdrop *>;  // view in device memory of superdroplets
using viewd_constsupers =
    Kokkos::View<const Superdrop *>;  // view in device memory of const superdroplets

using kkpair_size_t = Kokkos::pair<size_t, size_t>;  // kokkos pair of size_t (see supersingbx refs)
using subviewd_supers =
    Kokkos::Subview<viewd_supers, kkpair_size_t>;  // subiew of supers (for instance in a gridbox)
using subviewd_constsupers =
    Kokkos::Subview<viewd_constsupers,
                    kkpair_size_t>;  // const supers subview (for instance in a gridbox)

using mirrorh_constsupers = subviewd_constsupers::HostMirror;  // mirror view (copy) of subview of
                                                               // superdroplets on host memory

/* Random Number Generation */
using GenRandomPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;  // type for pool of thread safe
                                                                  // random number generators

/* Nested Parallelism */
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using TeamMember = TeamPolicy::member_type;

using HostTeamPolicy = Kokkos::TeamPolicy<HostSpace>;
using HostTeamMember = HostTeamPolicy::member_type;

#endif  // LIBS_SUPERDROPS_KOKKOSALIASES_SD_HPP_
