/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: createcartesianmaps.hpp
 * Project: cartesiandomain
 * Created Date: Wednesday 1st November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 9th November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * functions for creating a cartesian maps struct
 * from a GbxBoundsFromBinary struct containing
 * vectors of gridbox's indexes and their
 * coordinate (upper and lower) boundaries
 */

#ifndef LIBS_CARTESIANDOMAIN_CREATECARTESIANMAPS_HPP_
#define LIBS_CARTESIANDOMAIN_CREATECARTESIANMAPS_HPP_

#include <stdexcept>
#include <string_view>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

#include "../cleoconstants.hpp"
#include "../kokkosaliases.hpp"
#include "cartesiandomain/cartesianboundaryconds.hpp"
#include "cartesiandomain/cartesianmaps.hpp"
#include "initialise/gbxbounds_frombinary.hpp"

/* creates cartesian maps instance using gridbox bounds read from
gridfile for a 0-D, 1-D, 2-D or 3-D model with periodic or finite
boundary conditions (see note above). In a non-3D case, boundaries
and neighbours maps for unused dimensions are 'null'
(ie. return numerical limits), however the area and volume of each
gridbox remains finite. E.g. In the 0-D case, the bounds maps all
have 1 {key, value} where key=gbxidx=0 and value = {max, min}
numerical limits, meanwhile volume function returns a value determined
from the gridfile 'grid_filename' */
CartesianMaps create_cartesian_maps(const unsigned int ngbxs, const unsigned int nspacedims,
                                    std::string_view grid_filename);

#endif  // LIBS_CARTESIANDOMAIN_CREATECARTESIANMAPS_HPP_
