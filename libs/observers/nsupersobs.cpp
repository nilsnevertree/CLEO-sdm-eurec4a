/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: nsupersobs.cpp
 * Project: observers
 * Created Date: Thursday 30th November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 14th December 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * functionality to output nsupers
 * (per gridbox or total in domain)
 * to array in a zarr file system storage
 */

#include "observers/nsupersobs.hpp"

/* returns count of number of "raindrop-like" superdrops
for each gridbox. "raindrop-like" means radius > rlim.
  * WARNING! * When using OpenMP (supers in Host Space)
 and there are only a few superdroplets in supers,
 calc_nrainsupers is much slower then calc_nrainsupers_serial
 (probably because opening threads is more costly than the
 time saved in a parallel calculation over few elements) */
size_t calc_nrainsupers(const SupersInGbx &supersingbx) {
  constexpr double rlim(40e-6 / dlc::R0);  // dimless minimum radius of raindrop
  const subviewd_constsupers supers(supersingbx.readonly());
  const size_t nsupers(supers.extent(0));

  size_t nrainsupers(0);
  Kokkos::parallel_reduce(
      "calc_nrainsupers", Kokkos::RangePolicy<ExecSpace>(0, nsupers),
      KOKKOS_LAMBDA(const size_t kk, size_t &nrain) {
        const auto radius =
            supers(kk).get_radius();  // cast multiplicity from unsigned int to double
        if (radius >= rlim) {
          ++nrain;
        }
      },
      nrainsupers);

  return nrainsupers;
}

/* deep copy if necessary (if superdrops are on device not
  host memory), then returns count of number of "raindrop-like"
  superdrops for each gridbox. "raindrop-like" means radius > rlim */
size_t calc_nrainsupers_serial(const SupersInGbx &supersingbx) {
  constexpr double rlim(40e-6 / dlc::R0);  // dimless minimum radius of raindrop

  const auto h_supers = supersingbx.hostcopy();

  size_t nrainsupers(0);
  for (size_t kk(0); kk < h_supers.extent(0); ++kk) {
    const auto radius = h_supers(kk).get_radius();
    if (radius >= rlim) {
      ++nrainsupers;
    }
  }

  return nrainsupers;
}

void DoNrainsupersObs::nrainsupers_to_storage(const Gridbox &gbx) const {
  const size_t nrain(calc_nrainsupers(gbx.supersingbx));
  zarr->value_to_storage(nrain);
}
