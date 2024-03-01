/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: coalescence.hpp
 * Project: superdrops
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Wednesday 24th January 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * class and function to enact collision-coalescence events
 * in superdroplet model according to Shima et al. 2009.
 * Coalescence struct satisfies PairEnactX concept
 * used in Collisions struct
 */

#ifndef LIBS_SUPERDROPS_COALESCENCE_HPP_
#define LIBS_SUPERDROPS_COALESCENCE_HPP_

#include <cassert>
#include <functional>

#include <Kokkos_Core.hpp>

#include "./collisions.hpp"
#include "./microphysicalprocess.hpp"
#include "./nullsuperdrops.hpp"
#include "./superdrop.hpp"

struct DoCoalescence {
 private:
  /* if xi1 = gamma*xi2 coalescence makes twin SDs
  with same xi, r and solute mass. According to Shima et al. 2009
  Section 5.1.3. part (5) option (b)  */
  KOKKOS_FUNCTION void twin_superdroplet_coalescence(const uint64_t gamma, Superdrop &drop1,
                                                     Superdrop &drop2) const;

  /* if xi1 > gamma*xi2 coalescence grows sd2 radius and mass
  via decreasing multiplicity of sd1. According to
  Shima et al. 2009 Section 5.1.3. part (5) option (a)  */
  KOKKOS_FUNCTION void different_superdroplet_coalescence(const uint64_t gamma, Superdrop &drop1,
                                                          Superdrop &drop2) const;

 public:
  /* this operator is used as an "adaptor" for using
  DoCoalescence as a function in DoCollisions that
  satistfies the PairEnactX concept */
  KOKKOS_FUNCTION
  bool operator()(Superdrop &drop1, Superdrop &drop2, const double prob, const double phi) const;

  /* calculates value of gamma factor in Monte Carlo
  collision-coalescence as in Shima et al. 2009 */
  KOKKOS_FUNCTION uint64_t coalescence_gamma(const uint64_t xi1, const uint64_t xi2,
                                             const double prob, const double phi) const;

  /* coalesce pair of superdroplets by changing multiplicity,
  radius and solute mass of each superdroplet in pair
  according to Shima et al. 2009 Section 5.1.3. part (5) */
  KOKKOS_FUNCTION bool coalesce_superdroplet_pair(const uint64_t gamma, Superdrop &drop1,
                                                  Superdrop &drop2) const;
};

/* constructs Microphysical Process for collision-coalescence
of superdroplets with a constant timestep 'interval' and
probability of collision-coalescence determined by 'collcoalprob' */
template <PairProbability Probability>
inline MicrophysicalProcess auto CollCoal(const unsigned int interval,
                                          const std::function<double(unsigned int)> int2realtime,
                                          const Probability collcoalprob) {
  const auto DELT = int2realtime(interval);

  const DoCoalescence coal{};
  const DoCollisions<Probability, DoCoalescence> colls(DELT, collcoalprob, coal);
  return ConstTstepMicrophysics(interval, colls);
}

/* -----  ----- TODO: move functions below to .cpp file ----- ----- */

/* this operator is used as an "adaptor" for using
DoCoalescence as a function in DoCollisions that
satistfies the PairEnactX concept */
KOKKOS_FUNCTION bool DoCoalescence::operator()(Superdrop &drop1, Superdrop &drop2,
                                               const double prob, const double phi) const {
  /* 1. calculate gamma factor for collision-coalescence  */
  const auto xi1 = drop1.get_xi();
  const auto xi2 = drop2.get_xi();
  const auto gamma = coalescence_gamma(xi1, xi2, prob, phi);

  /* 2. enact collision-coalescence on pair
  of superdroplets if gamma is not zero */
  if (gamma != 0) {
    return coalesce_superdroplet_pair(gamma, drop1, drop2);
  }

  return 0;
}

/* calculates value of gamma factor in Monte Carlo
collision-coalescence as in Shima et al. 2009 */
KOKKOS_FUNCTION uint64_t DoCoalescence::coalescence_gamma(const uint64_t xi1, const uint64_t xi2,
                                                          const double prob,
                                                          const double phi) const {
  uint64_t gamma = floor(prob);  // if phi >= (prob - floor(prob))
  if (phi < (prob - gamma)) {
    ++gamma;
  }

  const auto maxgamma = xi1 / xi2;  // same as floor() for positive ints

  return Kokkos::fmin(gamma, maxgamma);
}

/* coalesce pair of superdroplets by changing multiplicity,
radius and solute mass of each superdroplet in pair
according to Shima et al. 2009 Section 5.1.3. part (5) */
KOKKOS_FUNCTION bool DoCoalescence::coalesce_superdroplet_pair(const uint64_t gamma,
                                                               Superdrop &drop1,
                                                               Superdrop &drop2) const {
  const auto xi1 = drop1.get_xi();
  const auto xi2 = drop2.get_xi();

  if (xi1 - gamma * xi2 > 0) {
    different_superdroplet_coalescence(gamma, drop1, drop2);
    return 0;
  } else if (xi1 - gamma * xi2 == 0) {
    twin_superdroplet_coalescence(gamma, drop1, drop2);

    /* if xi1 = xi2 = 1 before coalesence, then xi1=0 now */
    return is_null_superdrop(drop1);
    // return if_null_superdrop(drop1);
  }

  assert((xi1 >= gamma * xi2) &&
         "something undefined occured "
         "during colllision-coalescence");
  return 0;
}

/* if xi1 = gamma*xi2 coalescence makes twin SDs
with same xi, r and solute mass. According to Shima et al. 2009
Section 5.1.3. part (5) option (b). In rare case where
xi1 = xi2 = gamma = 1, new_xi of drop1 = 0 and drop1 should be removed
from domain.
Note: implicit casting of gamma (i.e. therefore droplets'
xi values) from uint64_t to double. */
KOKKOS_FUNCTION void DoCoalescence::twin_superdroplet_coalescence(const uint64_t gamma,
                                                                  Superdrop &drop1,
                                                                  Superdrop &drop2) const {
  const auto old_xi = drop2.get_xi();  // = drop1.xi
  const auto new_xi = old_xi / 2;      // same as floor() for positive ints

  assert((new_xi < old_xi) && "coalescence must decrease multiplicity");

  const auto new_rcubed = double{drop2.rcubed() + gamma * drop1.rcubed()};
  const auto new_r = double{Kokkos::pow(new_rcubed, (1.0 / 3.0))};

  const auto new_msol = double{drop2.get_msol() + gamma * drop1.get_msol()};

  drop1.set_xi(new_xi);
  drop2.set_xi(old_xi - new_xi);

  drop1.set_radius(new_r);
  drop2.set_radius(new_r);

  drop1.set_msol(new_msol);
  drop2.set_msol(new_msol);
}

/* if xi1 > gamma*xi2 coalescence grows drop2 radius and mass
via decreasing multiplicity of drop1. According to
Shima et al. 2009 Section 5.1.3. part (5) option (a)
Note: implicit casting of gamma (i.e. therefore droplets'
xi values) from uint64_t to double. */
KOKKOS_FUNCTION void DoCoalescence::different_superdroplet_coalescence(const uint64_t gamma,
                                                                       Superdrop &drop1,
                                                                       Superdrop &drop2) const {
  const auto new_xi = drop1.get_xi() - gamma * drop2.get_xi();

  assert((new_xi < drop1.get_xi()) && "coalescence must decrease multiplicity");

  drop1.set_xi(new_xi);

  const auto new_rcubed = double{drop2.rcubed() + gamma * drop1.rcubed()};

  drop2.set_radius(Kokkos::pow(new_rcubed, (1.0 / 3.0)));
  drop2.set_msol(drop2.get_msol() + gamma * drop1.get_msol());
}

#endif  // LIBS_SUPERDROPS_COALESCENCE_HPP_
