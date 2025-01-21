/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: createsupers.hpp
 * Project: runcleo
 * Created Date: Tuesday 17th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Friday 19th April 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * classes and templated functions required by RunCLEO to create a view of superdroplets
 * (on device) using some initial conditions
 */

#ifndef LIBS_RUNCLEO_CREATESUPERS_HPP_
#define LIBS_RUNCLEO_CREATESUPERS_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../kokkosaliases.hpp"
#include "gridboxes/sortsupers.hpp"
#include "runcleo/gensuperdrop.hpp"

/**
 * @brief Return an initialised view of superdrops in device memory.
 *
 * This function initialises a view of superdrops in device memory by creating
 * a view on the device and copying a host mirror view that is initialised using
 * the `SuperdropInitConds` instance.
 *
 * @tparam SuperdropInitConds The type of the super-droplets' initial conditions data.
 * @param sdic The instance of the super-droplets' initial conditions data.
 * @return A view of superdrops in device memory.
 */
template <typename SuperdropInitConds>
viewd_supers initialise_supers(const SuperdropInitConds &sdic);

/**
 * @brief Return a mirror view of superdrops on host memory.
 *
 * This function initialises a mirror view of superdrops on host memory, using the super-droplet
 * generator instance `GenSuperdrop` to generate the kk'th super-droplet with their initial Gridbox
 * index, spatial coordinates, and attributes. Kokkos::parallel_for is used to perform parallel
 * initialisation of the mirror view, where each superdrop is generated by the provided `gen`
 * function object.
 *
 * The equivalent serial version of the Kokkos::parallel_for([...]) is:
 * @code
 * for (size_t kk(0); kk < totnsupers; ++kk)
 * {
 *  h_supers(kk) = gen(kk);
 * }
 * @endcode
 *
 * @param sdic The instance of the super-droplets' initial conditions data.
 * @param supers The view of superdrops on device memory.
 * @return A mirror view of superdrops on host memory.
 */
template <typename SuperdropInitConds>
viewd_supers::HostMirror initialise_supers_on_host(const SuperdropInitConds &sdic,
                                                   const viewd_supers supers);

/**
 * @brief Check if superdroplets initialisation is complete.
 *
 * This function checks if the initialisation of supers view is complete by checking if the
 * superdroplets are sorted by ascending gridbox indexes. If the initialisation is incomplete
 * (the superdroplets are not sorted), it throws an exception with an appropriate error message.
 *
 * @param supers The view of super-droplets in device memory.
 *
 * @throws std::invalid_argument If the initialisation is incomplete i.e. the super-droplets
 * are not ordered correctly.
 */
void is_sdsinit_complete(const viewd_constsupers supers);

/**
 * @brief Print statement about initialised super-droplets.
 *
 * This function prints information about each superdroplet, including its ID, Gridbox index,
 * spatial coordinates, and attributes.
 *
 * @param supers The view of super-droplets in device memory.
 */
void print_supers(const viewd_constsupers supers);

/**
 * @brief Create a view of super-droplets in (device) memory.
 *
 * This function creates an ordered view of superdrops in device memory, where the number
 * of superdrops is specified by the parameter `totnsupers`. The superdrops are
 * ordered by the gridbox indexes and generated using a generator which uses
 * the initial conditions provided by the `SuperdropInitConds` type.
 *
 * Kokkos::Profiling are null pointers unless a Kokkos profiler library has been
 * exported to "KOKKOS_TOOLS_LIBS" prior to runtime so the lib gets dynamically loaded.
 *
 * @tparam SuperdropInitConds The type of the super-droplets' initial conditions data.
 * @param sdic The instance of the super-droplets' initial conditions data.
 * @return A view of super-droplets in device memory.
 */
template <typename SuperdropInitConds>
viewd_supers create_supers(const SuperdropInitConds &sdic) {
  Kokkos::Profiling::ScopedRegion region("init_supers");

  // Log message and create superdrops using the initial conditions
  std::cout << "\n--- create superdrops ---\ninitialising\n";
  viewd_supers supers(initialise_supers(sdic));

  // Log message and sort the view of superdrops
  std::cout << "sorting\n";
  supers = sort_supers(supers);

  // Log message and perform checks on the initialisation of superdrops
  std::cout << "checking initialisation\n";
  is_sdsinit_complete(supers);

  // // Print information about the created superdrops
  // print_supers(supers);

  // Log message indicating the successful creation of superdrops
  std::cout << "--- create superdrops: success ---\n";

  return supers;
}

/**
 * @brief Return an initialised view of superdrops in device memory.
 *
 * This function initialises a view of superdrops in device memory by creating
 * a view on the device and copying a host mirror view that is initialised using
 * the `SuperdropInitConds` instance.
 *
 * @tparam SuperdropInitConds The type of the super-droplets' initial conditions data.
 * @param sdic The instance of the super-droplets' initial conditions data.
 * @return A view of superdrops in device memory.
 */
template <typename SuperdropInitConds>
viewd_supers initialise_supers(const SuperdropInitConds &sdic) {
  GenSuperdrop gen(sdic);

  // create superdrops view on device
  viewd_supers supers("supers", gen.get_maxnsupers());

  // initialise a mirror of superdrops view on host
  auto h_supers = initialise_supers_on_host(sdic, supers);

  // Copy host view to device (h_supers to supers)
  Kokkos::deep_copy(supers, h_supers);

  return supers;
}

/**
 * @brief Return a mirror view of superdrops on host memory.
 *
 * This function initialises a mirror view of superdrops on host memory, using the super-droplet
 * generator instance `GenSuperdrop` to generate the kk'th super-droplet with their initial Gridbox
 * index, spatial coordinates, and attributes. Kokkos::parallel_for is used to perform parallel
 * initialisation of the mirror view, where each superdrop is generated by the provided `gen`
 * function object.
 *
 * The equivalent serial version of the Kokkos::parallel_for([...]) is:
 * @code
 * for (size_t kk(0); kk < totnsupers; ++kk)
 * {
 *  h_supers(kk) = gen(kk);
 * }
 * @endcode
 *
 * @param sdic The instance of the super-droplets' initial conditions data.
 * @param supers The view of superdrops on device memory.
 * @return A mirror view of superdrops on host memory.
 */
template <typename SuperdropInitConds>
viewd_supers::HostMirror initialise_supers_on_host(const SuperdropInitConds &sdic,
                                                   const viewd_supers supers) {
  // Create a mirror view of supers in case the original view is on device memory
  auto h_supers = Kokkos::create_mirror_view(supers);

  // Parallel initialisation of the mirror view
  const size_t totnsupers(supers.extent(0));
  const GenSuperdrop gen(sdic);
  Kokkos::parallel_for("initialise_supers_on_host", Kokkos::RangePolicy<HostSpace>(0, totnsupers),
                       [=](const size_t kk) { h_supers(kk) = gen(kk); });

  return h_supers;
}

#endif  // LIBS_RUNCLEO_CREATESUPERS_HPP_
