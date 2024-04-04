/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: superdrops_observer.hpp
 * Project: observers2
 * Created Date: Wednesday 24th January 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 4th April 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Observer to write variables related to superdroplet attributes at the start of
 * a constant interval timestep to ragged arrays in a dataset
 */

#ifndef LIBS_OBSERVERS2_SUPERDROPS_OBSERVER_HPP_
#define LIBS_OBSERVERS2_SUPERDROPS_OBSERVER_HPP_

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
#include "./write_to_dataset_observer.hpp"
#include "superdrops/superdrop.hpp"
#include "zarr2/dataset.hpp"

template <typename Store>
struct RaggedCount {
 private:
  std::shared_ptr<XarrayZarrArray<Store, uint32_t>> xzarr_ptr;  ///< pointer to raggedcount array

 public:
  RaggedCount(const Dataset<Store> &dataset, const size_t maxchunk)
      : xzarr_ptr(std::make_shared<XarrayZarrArray<Store, uint32_t>>(
            dataset.template create_raggedcount_array<uint32_t>(
                "raggedcount", "", "<u4", 1, {maxchunk}, {"time"}, "superdroplets"))) {}

  /* writes the total number of super-droplets in the domain "totnsupers" to the raggedcount array
  in the dataset. Note static conversion from architecture dependent, usually 16 byte unsigned
  integer (size_t = uint64_t), to 8 byte unsigned integer (uint32_t). */
  void write_to_array(const Dataset<Store> &dataset, const viewd_constsupers totsupers) const {
    const auto totnsupers = static_cast<uint32_t>(totsupers.extent(0));
    dataset.write_to_array(xzarr_ptr, totnsupers);
  }

  void write_arrayshape(const Dataset<Store> &dataset) const {
    dataset.write_arrayshape(xzarr_ptr);
  }
};

/* Operator is functor to perform copy of xi for each superdroplet in totsupers view to d_data
in parallel. Note conversion of xi from size_t (arch dependent usually 8 bytes) to long
precision unsigned int (unit64_t) */
struct XiFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t kk, viewd_constgbx d_gbxs, const viewd_constsupers totsupers,
                  Buffer<uint32_t>::mirrorviewd_buffer d_data) const {
    auto xi = static_cast<uint32_t>(totsupers(kk).get_xi());
    d_data(kk) = xi;
  }
};

/* Operator is functor to perform copy of radius for each superdroplet in totsupers view to d_data
in parallel. Note conversion of radius from double (8 bytes) to float (4 bytes) */
struct RadiusFunc {
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t kk, viewd_constgbx d_gbxs, const viewd_constsupers totsupers,
                  Buffer<float>::mirrorviewd_buffer d_data) const {
    auto radius = static_cast<float>(totsupers(kk).get_radius());
    d_data(kk) = radius;
  }
};

/* returns CollectDataForDataset which writes a variable (e.g. an attribute) from
each superdroplet to an array in a dataset in a given store for a given datatype and using a given
function-like functor */
template <typename Store, typename T, typename FunctorFunc>
CollectDataForDataset<Store> auto CollectSuperdropVariable(
    const Dataset<Store> &dataset, const FunctorFunc ffunc, const std::string_view name,
    const std::string_view units, const std::string_view dtype, const double scale_factor,
    const size_t maxchunk) {
  const auto chunkshape = std::vector<size_t>{maxchunk};
  const auto dimnames = std::vector<std::string>{"superdroplets"};
  const auto sampledimname = std::string_view("superdroplets");
  const auto xzarr = dataset.template create_ragged_array<T>(name, units, dtype, scale_factor,
                                                             chunkshape, dimnames, sampledimname);

  return GenericCollectData(ffunc, xzarr, 0);
}

template <typename Store>
CollectDataForDataset<Store> auto CollectXi(const Dataset<Store> &dataset, const int maxchunk) {
  return CollectSuperdropVariable<Store, uint32_t, XiFunc>(dataset, XiFunc{}, "xi", "", "<u4", 1,
                                                           maxchunk);
}

template <typename Store>
CollectDataForDataset<Store> auto CollectRadius(const Dataset<Store> &dataset, const int maxchunk) {
  return CollectSuperdropVariable<Store, float, RadiusFunc>(
      dataset, RadiusFunc{}, "radius", "micro-m", "<f4", dlc::R0 * 1e6, maxchunk);
}

/* constructs observer which writes writes superdroplet variables (e.g. an attributes) from each
superdroplet with a constant timestep 'interval' using an instance of the WriteToDatasetObserver
class given data collect struct "collect_data" */
template <typename Store>
inline Observer auto SuperdropsObserver(const unsigned int interval, const Dataset<Store> &dataset,
                                        const int maxchunk,
                                        CollectDataForDataset<Store> auto collect_data) {
  const CollectRaggedCount<Store> auto ragged_count = RaggedCount(dataset, maxchunk);
  return WriteToDatasetObserver(interval, dataset, collect_data, ragged_count);
}

#endif  // LIBS_OBSERVERS2_SUPERDROPS_OBSERVER_HPP_
