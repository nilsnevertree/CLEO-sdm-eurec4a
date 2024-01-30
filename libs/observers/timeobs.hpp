/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: timeobs.hpp
 * Project: observers
 * Created Date: Friday 20th October 2023
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
 * Observer to output gbxindx to array in a
 * zarr file system storage
 */

#ifndef LIBS_OBSERVERS_TIMEOBS_HPP_
#define LIBS_OBSERVERS_TIMEOBS_HPP_

#include <concepts>
#include <functional>
#include <iostream>
#include <memory>

#include <Kokkos_Core.hpp>

#include "../cleoconstants.hpp"
#include "../kokkosaliases.hpp"
#include "./observers.hpp"
#include "gridboxes/gridbox.hpp"
#include "zarr/coordstorage.hpp"

namespace dlc = dimless_constants;

/* constructs observer of time with a
constant timestep 'interval' using an
instance of the DoTimeObs class */
inline Observer auto TimeObserver(const unsigned int interval, FSStore &store, const int maxchunk,
                                  const std::function<double(unsigned int)> step2dimlesstime);

/* observe time of 0th gridbox and write it
to an array 'zarr' store as determined by
the CoordStorage instance */
class DoTimeObs {
 private:
  using store_type = CoordStorage<double>;
  std::shared_ptr<store_type> zarr;
  std::function<double(unsigned int)>
      step2dimlesstime;  // function to convert timesteps to real time

 public:
  DoTimeObs(FSStore &store, const int maxchunk,
            const std::function<double(unsigned int)> step2dimlesstime)
      : zarr(std::make_shared<store_type>(store, maxchunk, "time", "<f8", "s", dlc::TIME0)),
        step2dimlesstime(step2dimlesstime) {
    zarr->is_name("time");
  }

  void before_timestepping(const viewh_constgbx h_gbxs) const {
    std::cout << "observer includes TimeObserver\n";
  }

  void after_timestepping() const {}

  void at_start_step(const unsigned int t_mdl, const viewh_constgbx h_gbxs,
                     const viewd_constsupers totsupers) const {
    at_start_step(t_mdl);
  }

  /* converts integer model timestep to dimensionless time,
  then writes to zarr coordinate storage */
  void at_start_step(const unsigned int t_mdl) const {
    const auto time = step2dimlesstime(t_mdl);
    zarr->value_to_storage(time);
  }

  void at_start_step(const unsigned int t_mdl, const Gridbox &gbx) const {}
};

/* constructs observer of time with a
constant timestep 'interval' using an
instance of the DoTimeObs class */
inline Observer auto TimeObserver(const unsigned int interval, FSStore &store, const int maxchunk,
                                  const std::function<double(unsigned int)> step2dimlesstime) {
  const auto obs = DoTimeObs(store, maxchunk, step2dimlesstime);
  return ConstTstepObserver(interval, obs);
}

#endif  // LIBS_OBSERVERS_TIMEOBS_HPP_
