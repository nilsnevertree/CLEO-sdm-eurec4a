/*
 * ----- CLEO -----
 * File: cartesianmotion.cpp
 * Project: cartesiandomain
 * Created Date: Wednesday 8th November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 9th November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 */

#include "./cartesianmotion.hpp"

KOKKOS_FUNCTION void
cartesian_update_superdrop_gbxindex(const unsigned int gbxindex,
                                    const CartesianMaps &gbxmaps,
                                    Superdrop &drop)
/* Updates superdroplet sdgbxindex in a cartesian domain.
For each direction (z, then x, then y), gbxmaps's forward and backward
get_neighbour functions are passed into update_superdrop_ifneighbour
along with superdroplet and the gridbox bounds for that direction.
(If coord not within bounds, update_superdrop_ifneighbour should call
appropriate get_neighbour function to update the superdroplet's
sd_gbxindex (and possibly other attributes if desired). After algorithm
for z, then x, then y directions are complete, resultant sd_gbxindex
is returned. */
{
  unsigned int current_gbxindex(gbxindex);

  current_gbxindex = move_to_coord3neighbour(gbxmaps, gbxindex, drop);

  // current_gbxindex = update_ifneighbour(
  //     gbxmaps, zdown, zup,
  //     [](const Maps4GridBoxes &gbxmaps, const auto ii)
  //     { return gbxmaps.get_bounds_z(ii); },
  //     [](const Superdrop &superdrop)
  //     { return superdrop.coord3; },
  //     current_gbxindex, superdrop);

  // current_gbxindex = update_ifneighbour(
  //     gbxmaps, xbehind, xinfront,
  //     [](const Maps4GridBoxes &gbxmaps,
  //        const unsigned int ii)
  //     { return gbxmaps.get_bounds_x(ii); },
  //     [](const Superdrop &superdrop)
  //     { return superdrop.coord1; },
  //     current_gbxindex, superdrop);

  // current_gbxindex = update_ifneighbour(
  //     gbxmaps, yleft, yright,
  //     [](const Maps4GridBoxes &gbxmaps,
  //        const unsigned int ii)
  //     { return gbxmaps.get_bounds_y(ii); },
  //     [](const Superdrop &superdrop)
  //     { return superdrop.coord2; },
  //     current_gbxindex, superdrop);

  drop.set_sdgbxindex(current_gbxindex);
}

KOKKOS_FUNCTION void
check_inbounds_or_outdomain(const unsigned int current_gbxindex,
                            const Kokkos::pair<double, double> bounds,
                            const double coord) const
/* raise error if superdrop not out of domain or within bounds */
{
  const bool bad_gbxindex((current_gbxindex != LIMITVALUES::uintmax) &&
                          ((coord < bounds.first) | (coord >= bounds.second)));

  assert((!bad_gbxindex) && "SD not in previous gbx nor a neighbour."
                            " Try reducing the motion timestep to"
                            " satisfy CFL criteria, or use "
                            " 'update_ifoutside' to update sd_gbxindex");
}

int is_change_sdgbxinde(const unsigned int current_gbxindex,
                        const Kokkos::pair<double, double> bounds,
                        const double coord)
/* Given bounds = {lowerbound, upperbound} of a gridbox with
index 'gbxindex', function determines if coord is within bounds
of that gridbox. (Note: lower bound inclusive, upper bound exclusive).
If coord not within bounds backwardsidx or forwardsidx function,
as appropriate, is used to return a neighbouring gridbox's index.
If coord lies within bounds, gbxindex is returned. If index is
already out of domain (ie. value is the maximum unsigned int),
return out of domain index */
{
  if (current_gbxindex == LIMITVALUES::uintmax)
  {
    return 0; // ignore current_gbxindex that is already out of domain
  }
  else if (coord < bounds.first) // lowerbound
  {
    return 1; // current_gbxindex -> backwards_neighbour
  }
  else if (coord >= bounds.second) // upperbound
  {
    return 2; // current_gbxindex -> forwards_neighbour
  }
  else
  {
    return 0; // no change to current_gbxindex if coord within bounds
  }
}

KOKKOS_FUNCTION unsigned int
move_to_coord3neighbour(const CartesianMaps &gbxmaps,
                        unsigned int current_gbxindex,
                        Superdrop &drop)
/* For a given direction, pass {lower, upper} bounds into
update_superdrop_ifneighbour to get updated gbxindex and superdrop
(e.g. if superdroplet's coord fromsdcoord function lies outside of
bounds given gbxbounds using current_gbxindex). Repeat until
superdroplet coord is within the bounds given by the current_gbxindex,
or until superdrop leaves domain. */
{
  const int val(is_change_sdgbxindex(current_gbxindex,
                                     gbxmaps.coord3bounds(current_gbxindex),
                                     drop.get_coord3())); // value != 0 if sdgbxindex needs to change

  switch (val)
  {
  case 1:
    // current_gbxindex = backwards_neighbour_z(gbxmaps, current_gbxindex, superdrop);
    break;
  case 2:
    //  current_gbxindex = forwards_neighbour_z(gbxmaps, current_gbxindex, superdrop);
    break;
  }

  check_inbounds_or_outdomain(current_gbxindex,
                              gbxmaps.coord3bounds(current_gbxindex),
                              drop.get_coord3());

  return current_gbxindex;
}