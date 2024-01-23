/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: initialconditions.hpp
 * Project: runcleo
 * Created Date: Tuesday 17th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Friday 3rd November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * concept and struct obeying concept for
 * use in initial conditions of superdroplets
 * and gridboxes in CLEO SDM
 */

#ifndef LIBS_RUNCLEO_INITIALCONDITIONS_HPP_
#define LIBS_RUNCLEO_INITIALCONDITIONS_HPP_

#include <concepts>
#include <utility>
#include <vector>

#include "../kokkosaliases.hpp"
#include "initialise/initconds.hpp"

/* concept InitialConditions is all types that have
initsupers and initgbxs structs which can call functions listed */
template <typename IC>
concept InitialConditions =
    requires(IC ic, unsigned int t, const viewh_constgbx h_gbxs, InitSupersData initdata) {
      { ic.initsupers.get_nspacedims() } -> std::convertible_to<unsigned int>;
      { ic.initsupers.get_totnsupers() } -> std::convertible_to<size_t>;
      { ic.initsupers.fetch_data_size() } -> std::convertible_to<size_t>;
      { ic.initsupers.fetch_data(initdata) } -> std::same_as<void>;

      { ic.initgbxs.get_ngbxs() } -> std::convertible_to<size_t>;
      { ic.initgbxs.press() } -> std::convertible_to<std::vector<double>>;
      { ic.initgbxs.temp() } -> std::convertible_to<std::vector<double>>;
      { ic.initgbxs.qvap() } -> std::convertible_to<std::vector<double>>;
      { ic.initgbxs.qcond() } -> std::convertible_to<std::vector<double>>;
      { ic.initgbxs.wvel() } -> std::convertible_to<std::vector<std::pair<double, double>>>;
      { ic.initgbxs.uvel() } -> std::convertible_to<std::vector<std::pair<double, double>>>;
      { ic.initgbxs.vvel() } -> std::convertible_to<std::vector<std::pair<double, double>>>;
    };

#endif  // LIBS_RUNCLEO_INITIALCONDITIONS_HPP_
