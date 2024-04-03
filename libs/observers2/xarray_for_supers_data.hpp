/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: xarray_for_supers_data.hpp
 * Project: observers2
 * Created Date: Wednesday 24th January 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Wednesday 3rd April 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Helpful structs to write data collected from each superdroplet in parallel
 * to a ragged array in a dataset
 */

#ifndef LIBS_OBSERVERS2_XARRAY_FOR_SUPERS_DATA_HPP_
#define LIBS_OBSERVERS2_XARRAY_FOR_SUPERS_DATA_HPP_

#include <Kokkos_Core.hpp>
#include <concepts>
#include <string_view>

#include "../kokkosaliases.hpp"
#include "superdrops/superdrop.hpp"
#include "zarr2/buffer.hpp"
#include "zarr2/dataset.hpp"
#include "zarr2/xarray_zarr_array.hpp"

/* struct holding an array in a dataset as well a view and its mirror view
which can be useful when collecting data for 1 variable from 'totnsupers' superdroplets
(in parallel) to then writing to the ragged array */
template <typename Store, typename T>
struct XarrayForSupersData {
  using viewh_data = Buffer<T>::viewh_buffer;              // type of view for h_data
  using mirrorviewd_data = Buffer<T>::mirrorviewd_buffer;  // mirror view type for d_data
  XarrayZarrArray<Store, T> xzarr;                         // array in a dataset
  viewh_data h_data;        // view on host for value of 1 variable from every superdrop
  mirrorviewd_data d_data;  // mirror view of h_data on device

  /* Constructor to initialize views and pointer to array in dataset */
  XarrayForSupersData(const Dataset<Store> &dataset, const std::string_view name,
                      const std::string_view units, const std::string_view dtype,
                      const double scale_factor, const size_t maxchunk)
      : xzarr(dataset.template create_ragged_array<T>(name, units, dtype, scale_factor, {maxchunk},
                                                      {"time"}, "superdroplets")),
        h_data("h_data", 0),
        d_data(Kokkos::create_mirror_view(ExecSpace(), h_data)) {}

  void reallocate_dataviews(const size_t totnsupers) {
    Kokkos::realloc(h_data, totnsupers);
    Kokkos::realloc(d_data, totnsupers);
  }

  /* copy data from device view directly to host and then write to array in dataset */
  void write_to_array(const Dataset<Store> &dataset) {
    Kokkos::deep_copy(h_data, d_data);
    dataset.write_to_array(xzarr, h_data);
  }

  /* call function to write shape of array according to dataset */
  void write_arrayshape(const Dataset<Store> &dataset) { dataset.write_arrayshape(xzarr); }
};

#endif  // LIBS_OBSERVERS2_XARRAY_FOR_SUPERS_DATA_HPP_
