/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: massmoments_observer.cpp
 * Project: observers
 * Created Date: Wednesday 24th January 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Wednesday 5th June 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Functionality to calculate mass moments of (rain)droplet
 * distribution in each gridbox in parallel
 */

#include "./massmoments_observer.hpp"

/**
 * @brief Perform calculation of 0th, 1st, and 2nd moments of the (real)
 * droplet mass distribution for a single gridbox through reduction over super-droplets.
 *
 * This operator is a functor to perform the calculation of the 0th, 1st, and 2nd moments
 * of the droplet mass distribution in a gridbox (i.e. 0th, 3rd, and 6th moments of the
 * droplet radius distribution) within a Kokkos::parallel_reduce range policy
 * loop over superdroplets within a team policy loop over gridboxes.
 *
 * Kokkos::parallel_reduce([...]) is equivalent in serial to sum over result of:
 * for (size_t kk(0); kk < supers.extent(0); ++kk){[...]}.
 *
 * _Note:_ conversion from 8 to 4-byte precision for all mass moments: mom0 from size_t
 * (architecture dependent usually long unsigned int = 8 bytes) to 8 byte unsigned integer, and
 * mom1 and mom2 from double (8 bytes) to float (4 bytes).
 *
 * @param team_member The Kokkos team member.
 * @param supers The view of super-droplets for a gridbox (on device).
 * @param d_mom0 The view for the 0th mass moment.
 * @param d_mom1 The view for the 1st mass moment.
 * @param d_mom2 The view for the 2nd mass moment.
 */
KOKKOS_FUNCTION
void calculate_massmoments(const TeamMember &team_member, const viewd_constsupers supers,
                           const auto d_mom0, const auto d_mom1, const auto d_mom2) {
  const size_t nsupers(supers.extent(0));

  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, nsupers),
      KOKKOS_LAMBDA(const size_t kk, uint64_t &m0, float &m1, float &m2) {
        const auto &drop(supers(kk));

        assert((drop.get_xi() < LIMITVALUES::uint64_t_max) &&
               "superdroplet mulitiplicy too large to represent with 4 byte unsigned integer");
        m0 += static_cast<uint64_t>(drop.get_xi());

        const auto mass = drop.mass();
        const auto xi = static_cast<double>(drop.get_xi());  // cast multiplicity to double
        m1 += static_cast<float>(xi * mass);
        m2 += static_cast<float>(xi * mass * mass);
      },
      d_mom0(ii), d_mom1(ii), d_mom2(ii));  // {0th, 1st, 2nd} mass moments
}

/**
 * @brief Functor operator to perform calculation of 0th, 1st, and 2nd moments of the (real)
 * droplet mass distribution in each gridbox.
 *
 * This operator is a functor to perform the calculation of the 0th, 1st, and 2nd moments
 * of the droplet mass distribution in each gridbox (i.e. 0th, 3rd, and 6th moments of the
 * droplet radius distribution) within a Kokkos::parallel_for range policy
 * loop over superdroplets.
 *
 * A raindroplet is a droplet with a radius >= rlim = 40microns.
 *
 * Kokkos::parallel_reduce([...]) is equivalent in serial to sum over result of:
 * for (size_t kk(0); kk < supers.extent(0); ++kk){[...]}.
 *
 * _Note:_ conversion from 8 to 4-byte precision for all mass moments: mom0 from size_t
 * (architecture dependent usually long unsigned int = 8 bytes) to 8 byte unsigned integer, and
 * mom1 and mom2 from double (8 bytes) to float (4 bytes).
 *
 * @param team_member The Kokkos team member.
 * @param d_gbxs The view of gridboxes on device.
 * @param d_mom0 The mirror view buffer for the 0th mass moment.
 * @param d_mom1 The mirror view buffer for the 1st mass moment.
 * @param d_mom2 The mirror view buffer for the 2nd mass moment.
 */
KOKKOS_FUNCTION
void RaindropsMassMomentsFunc::operator()(const TeamMember &team_member,
                                          const viewd_constgbx d_gbxs,
                                          Buffer<uint64_t>::mirrorviewd_buffer d_mom0,
                                          Buffer<float>::mirrorviewd_buffer d_mom1,
                                          Buffer<float>::mirrorviewd_buffer d_mom2) const {
  constexpr double rlim(40e-6 / dlc::R0);  // dimless minimum radius of raindrop
  const auto ii = team_member.league_rank();
  const auto supers(d_gbxs(ii).supersingbx.readonly());

  const size_t nsupers(supers.extent(0));
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, nsupers),
      KOKKOS_LAMBDA(const size_t kk, uint64_t &m0, float &m1, float &m2) {
        const auto &drop(supers(kk));
        const auto binary = bool{drop.get_radius() >= rlim};  // 1 if droplet is raindrop, else 0

        assert((drop.get_xi() < LIMITVALUES::uint64_t_max) &&
               "superdroplet mulitiplicy too large to represent with 4 byte unsigned integer");
        m0 += static_cast<uint64_t>(drop.get_xi() * binary);

        const auto mass = drop.mass();
        const auto xi = static_cast<double>(drop.get_xi());  // cast multiplicity to double
        m1 += static_cast<float>(xi * mass * binary);
        m2 += static_cast<float>(xi * mass * mass * binary);
      },
      d_mom0(ii), d_mom1(ii), d_mom2(ii));  // {0th, 1st, 2nd} mass moments
}
