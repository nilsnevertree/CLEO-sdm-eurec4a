/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: state_writers.hpp
 * Project: observers2
 * Created Date: Wednesday 24th January 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Friday 29th March 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Functions to create WriteGridboxToArrays which write out state variables from
 * each Gridbox, e.g. to use in StateObserver.
 */

#ifndef LIBS_OBSERVERS2_STATE_WRITERS_HPP_
#define LIBS_OBSERVERS2_STATE_WRITERS_HPP_

#include <Kokkos_Core.hpp>
#include <concepts>
#include <memory>
#include <string_view>

#include "../cleoconstants.hpp"
#include "../kokkosaliases.hpp"
#include "./write_gridbox_to_array.hpp"
#include "gridboxes/gridbox.hpp"
#include "superdrops/state.hpp"
#include "zarr2/dataset.hpp"
#include "zarr2/xarray_zarr_array.hpp"
#include "zarr2/zarr_array.hpp"

// Operator is functor to perform copy of pressure in each gridbox to d_data in parallel.
// Note conversion of pressure from double (8 bytes) to single precision (4 bytes float) in output
struct PressFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t ii, viewd_constgbx d_gbxs,
                  Buffer<float>::mirrorviewd_buffer d_data) const {
    auto press = static_cast<float>(d_gbxs(ii).state.press);
    d_data(ii) = press;
  }
};

// Operator is functor to perform copy of temperature in each gridbox to d_data in parallel.
// Note conversion of temperature from double (8 bytes) to single precision (4 bytes float) in
// output
struct TempFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t ii, viewd_constgbx d_gbxs,
                  Buffer<float>::mirrorviewd_buffer d_data) const {
    auto temp = static_cast<float>(d_gbxs(ii).state.temp);
    d_data(ii) = temp;
  }
};

// Operator is functor to perform copy of vapour mass mixing ratio (qvap) in each gridbox to d_data
// in parallel. Note conversion of qvap from double (8 bytes) to single precision (4 bytes float) in
// output
struct QvapFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t ii, viewd_constgbx d_gbxs,
                  Buffer<float>::mirrorviewd_buffer d_data) const {
    auto qvap = static_cast<float>(d_gbxs(ii).state.qvap);
    d_data(ii) = qvap;
  }
};

// Operator is functor to perform copy of liquid mass mixing ratio (qcond) in each gridbox to d_data
// in parallel. Note conversion of qcond from double (8 bytes) to single precision (4 bytes
// float) in output
struct QcondFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t ii, viewd_constgbx d_gbxs,
                  Buffer<float>::mirrorviewd_buffer d_data) const {
    auto qcond = static_cast<float>(d_gbxs(ii).state.qcond);
    d_data(ii) = qcond;
  }
};

// Operator is functor to perform copy of wvel at the centre of each gridbox to d_data
// in parallel. Note conversion of wvel from double (8 bytes) to single precision (4 bytes
// float) in output
struct WvelFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t ii, viewd_constgbx d_gbxs,
                  Buffer<float>::mirrorviewd_buffer d_data) const {
    auto wvel = static_cast<float>(d_gbxs(ii).state.wvelcentre());
    d_data(ii) = wvel;
  }
};

// Operator is functor to perform copy of uvel at the centre of each gridbox to d_data
// in parallel. Note conversion of uvel from double (8 bytes) to single precision (4 bytes
// float) in output
struct UvelFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t ii, viewd_constgbx d_gbxs,
                  Buffer<float>::mirrorviewd_buffer d_data) const {
    auto uvel = static_cast<float>(d_gbxs(ii).state.uvelcentre());
    d_data(ii) = uvel;
  }
};

// Operator is functor to perform copy of vvel at the centre of each gridbox to d_data
// in parallel. Note conversion of vvel from double (8 bytes) to single precision (4 bytes
// float) in output
struct VvelFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t ii, viewd_constgbx d_gbxs,
                  Buffer<float>::mirrorviewd_buffer d_data) const {
    auto vvel = static_cast<float>(d_gbxs(ii).state.vvelcentre());
    d_data(ii) = vvel;
  }
};

// returns WriteGridboxToArray which writes the pressure, temperature, qvap, and qcond from
// each gridbox to arrays in a dataset in a store
template <typename Store>
WriteGridboxToArray<Store> auto ThermoWriter(Dataset<Store> &dataset, const int maxchunk,
                                             const size_t ngbxs) {
  // create shared pointer to 2-D array in dataset for pressure in each gridbox over time
  auto press_ptr = std::make_shared<XarrayForGenericGbxWriter<Store, float>>(
      dataset, "press", "hPa", "<f4", dlc::P0 / 100, maxchunk, ngbxs);

  // create shared pointer to 2-D array in dataset for temperature in each gridbox over time
  auto temp_ptr = std::make_shared<XarrayForGenericGbxWriter<Store, float>>(
      dataset, "temp", "K", "<f4", dlc::TEMP0, maxchunk, ngbxs);

  // create shared pointer to 2-D array in dataset for qvap in each gridbox over time
  auto qvap_ptr = std::make_shared<XarrayForGenericGbxWriter<Store, float>>(
      dataset, "qvap", "g/Kg", "<f4", 1000.0, maxchunk, ngbxs);

  // create shared pointer to 2-D array in dataset for qcond in each gridbox over time
  auto qcond_ptr = std::make_shared<XarrayForGenericGbxWriter<Store, float>>(
      dataset, "qcond", "g/Kg", "<f4", 1000.0, maxchunk, ngbxs);

  const auto c = CombineGDW<Store>{};
  auto press = GenericGbxWriter<Store, float, PressFunc>(press_ptr, PressFunc{});
  auto temp = GenericGbxWriter<Store, float, TempFunc>(temp_ptr, TempFunc{});
  auto qvap = GenericGbxWriter<Store, float, QvapFunc>(qvap_ptr, QvapFunc{});
  auto qcond = GenericGbxWriter<Store, float, QcondFunc>(qcond_ptr, QcondFunc{});

  return c(c(qvap, c(press, temp)), qcond);
}

// returns WriteGridboxToArray which writes the wind velocity components from the centre of each
// gridbox to arrays in a dataset in a store
template <typename Store>
WriteGridboxToArray<Store> auto WindVelocityWriter(Dataset<Store> &dataset, const int maxchunk,
                                                   const size_t ngbxs) {
  auto vel_ptr =
      [&dataset, maxchunk, ngbxs](
          const std::string_view name) -> std::shared_ptr<XarrayForGenericGbxWriter<Store, float>> {
    return std::make_shared<XarrayForGenericGbxWriter<Store, float>>(dataset, name, "m/s", "<f4",
                                                                     dlc::W0, maxchunk, ngbxs);
  };
  // create shared pointer to 2-D arrays in a datasetfor the velocity at the centre of each gridbox
  // over time and use to make GbxWriters for each velocity component
  auto wvel = GenericGbxWriter<Store, float, WvelFunc>(vel_ptr("wvel"), WvelFunc{});
  auto uvel = GenericGbxWriter<Store, float, UvelFunc>(vel_ptr("uvel"), UvelFunc{});
  auto vvel = GenericGbxWriter<Store, float, VvelFunc>(vel_ptr("vvel"), VvelFunc{});

  const auto c = CombineGDW<Store>{};
  return c(wvel, c(vvel, uvel));
}

#endif  // LIBS_OBSERVERS2_STATE_WRITERS_HPP_
