/*
 * ----- CLEO -----
 * File: createsupers.hpp
 * Project: runcleo
 * Created Date: Tuesday 17th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 2nd November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * file for structure(s) to create a view of
 * superdroplets (on device) using some
 * initial conditions
 */

#ifndef CREATESUPERS_HPP
#define CREATESUPERS_HPP

#include <memory>
#include <array>
#include <iostream>
#include <string>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "../kokkosaliases.hpp"
#include "initialise/initconds.hpp"
#include "superdrops/superdrop.hpp"
#include "gridboxes/sortsupers.hpp"

template <typename SuperdropInitConds>
viewd_supers create_supers(const SuperdropInitConds &sdic);
/* create view of "totnsupers" number of superdrops
(in device memory) which is ordered by the superdrops'
gridbox indexes using the initial conditions
generated by the referenced SuperdropInitConds type */

class GenSuperdrop
/* struct holds vectors for data for the initial
conditions of some superdroplets' properties and
returns superdrops generated from them */
{
private:
  unsigned int nspacedims;
  std::shared_ptr<Superdrop::IDType::Gen> sdIdGen; // pointer to superdrop id generator
  InitSupersData initdata;

  std::array<double, 3> coords_at(const unsigned int kk) const;

  SuperdropAttrs attrs_at(const unsigned int kk) const;

public:
  template <typename SuperdropInitConds>
  GenSuperdrop(const SuperdropInitConds &sdic)
      : nspacedims(sdic.get_nspacedims()),
        sdIdGen(std::make_shared<Superdrop::IDType::Gen>())
        {
          sdic.fetch_data(initdata); 
        }

  Superdrop operator()(const unsigned int kk) const;
};

template <typename SuperdropInitConds>
viewd_supers initialise_supers(const SuperdropInitConds &sdic);
/* return an initialised view of superdrops on
device memory by copying a host mirror view that
is initialised using the SuperdropInitConds instance */

viewd_supers::HostMirror
initialise_supers_on_host(const GenSuperdrop &gen,
                          const viewd_supers supers);
/* return mirror view of superdrops (on host memory)
which have been initialised using data from a 
superdroplet generator 'gen' for their initial gbxindex,
spatial coordinates and attributes */

void is_sdsinit_complete(const viewd_constsupers supers,
                         const size_t size);
/* ensure the number of superdrops in the view matches the
size according to the initial conditions */

void print_supers(const viewd_constsupers supers);
/* print superdroplet information */

template <typename SuperdropInitConds>
viewd_supers create_supers(const SuperdropInitConds &sdic)
/* create view of "totnsupers" number of superdrops
(in device memory) which is ordered by the superdrops'
gridbox indexes using the initial conditions
generated by the referenced SuperdropInitConds type */
{
  std::cout << "\n--- create superdrops ---\n"
            << "initialising\n";
  viewd_supers supers(initialise_supers(sdic));

  std::cout << "sorting\n";
  supers = sort_supers(supers);

  std::cout << "checking initialisation\n";
  is_sdsinit_complete(supers, sdic.fetch_data_size());
  print_supers(supers);

  std::cout << "--- create superdrops: success ---\n";

  return supers;
}

template <typename SuperdropInitConds>
viewd_supers initialise_supers(const SuperdropInitConds &sdic)
/* return an initialised view of superdrops on
device memory by copying a host mirror view that
is initialised using the SuperdropInitConds instance */
{
  /* create superdrops view on device */
  viewd_supers supers("supers", sdic.get_totnsupers());

  /* initialise a mirror of supers view on host*/
  const GenSuperdrop gen(sdic);
  auto h_supers = initialise_supers_on_host(gen, supers);

  /* copy host view to device (h_supers to supers) */
  Kokkos::deep_copy(supers, h_supers);

  return supers;
}

viewd_supers::HostMirror
initialise_supers_on_host(const GenSuperdrop &gen,
                          const viewd_supers supers)
/* return mirror view of superdrops (on host memory)
which have been initialised using data from a
SuperdropInitConds instance for their initial gbxindex,
spatial coordinates and attributes.
Kokkos::parallel_for([...]) is equivalent to:
for (size_t kk(0); kk < totnsupers; ++kk) {[...]}
when in serial */
{
  const size_t totnsupers(supers.extent(0));

  /* equivalent serial version of parallel_for loop below
  for (size_t kk(0); kk < totnsupers; ++kk)
  {
    h_supers(kk) = gen(kk); 
  }
  */

  auto h_supers = Kokkos::create_mirror_view(supers); // mirror of supers in case view is on device memory
  Kokkos::parallel_for(
      "initialise_supers_on_host",
      Kokkos::RangePolicy<HostSpace>(0, totnsupers),
      [=](const size_t kk)
      {
        h_supers(kk) = gen(kk);
      });

  return h_supers;
}

#endif // CREATESUPERS_HPP