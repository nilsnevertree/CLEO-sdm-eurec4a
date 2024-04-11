/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: state_observer.hpp
 * Project: observers
 * Created Date: Wednesday 24th January 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 11th April 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Observer to write variables related to Gridboxes' state at the start of
 * a constant interval timestep to arrays in a dataset
 */

#ifndef LIBS_OBSERVERS_STATE_OBSERVER_HPP_
#define LIBS_OBSERVERS_STATE_OBSERVER_HPP_

#include <Kokkos_Core.hpp>
#include <concepts>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../cleoconstants.hpp"
#include "../kokkosaliases.hpp"
#include "./collect_data_for_dataset.hpp"
#include "./generic_collect_data.hpp"
#include "./thermo_observer.hpp"
#include "./windvel_observer.hpp"
#include "./write_to_dataset_observer.hpp"
#include "gridboxes/gridbox.hpp"
#include "zarr/dataset.hpp"

/**
 * @brief Constructs an observer which writes the state of a gridbox (thermodynamics and
 * wind velocity components) in each gridbox at start of each observation timestep to an array with
 * a constant observation timestep "interval".
 *
 * This function collects thermodynamic properties and wind velocities from the dataset and combines
 * them into a single collection of state data.
 *
 * @tparam Store Type of store for dataset.
 * @param interval Observation timestep.
 * @param dataset Dataset to write time data to.
 * @param maxchunk Maximum number of elements in a chunk (1-D vector size).
 * @param ngbxs The number of gridboxes.
 * @return Observer An observer instance for writing the state data.
 */
template <typename Store>
inline Observer auto StateObserver(const unsigned int interval, const Dataset<Store> &dataset,
                                   const int maxchunk, const size_t ngbxs) {
  const CollectDataForDataset<Store> auto thermo = CollectThermo(dataset, maxchunk, ngbxs);
  const CollectDataForDataset<Store> auto windvel = CollectWindVel(dataset, maxchunk, ngbxs);

  const CollectDataForDataset<Store> auto collect_data = windvel >> thermo;

  return WriteToDatasetObserver(interval, dataset, collect_data);
}

#endif  // LIBS_OBSERVERS_STATE_OBSERVER_HPP_
