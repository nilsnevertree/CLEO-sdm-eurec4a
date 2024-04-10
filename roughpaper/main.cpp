/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: main.cpp
 * Project: roughpaper
 * Created Date: Monday 29th January 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Monday 8th April 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * rough paper for checking small things
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <concepts>
#include <iostream>

#include "cartesiandomain/cartesianmaps.hpp"
#include "cartesiandomain/createcartesianmaps.hpp"
#include "coupldyn_fromfile/fromfile_cartesian_dynamics.hpp"
#include "coupldyn_fromfile/fromfilecomms.hpp"
#include "gridboxes/gridboxmaps.hpp"
#include "initialise/config.hpp"
#include "initialise/initgbxs_null.hpp"
#include "initialise/initsupers_frombinary.hpp"
#include "initialise/timesteps.hpp"
#include "observers2/gbxindex_observer.hpp"
#include "observers2/massmoments_observer.hpp"
#include "observers2/nsupers_observer.hpp"
#include "observers2/observers.hpp"
#include "observers2/runstats_observer.hpp"
#include "observers2/state_observer.hpp"
#include "observers2/streamout_observer.hpp"
#include "observers2/superdrops_observer.hpp"
#include "observers2/thermo_observer.hpp"
#include "observers2/time_observer.hpp"
#include "observers2/totnsupers_observer.hpp"
#include "observers2/windvel_observer.hpp"
#include "runcleo/coupleddynamics.hpp"
#include "runcleo/couplingcomms.hpp"
#include "runcleo/initialconditions.hpp"
#include "runcleo/runcleo.hpp"
#include "runcleo/sdmmethods.hpp"
#include "superdrops/microphysicalprocess.hpp"
#include "superdrops/motion.hpp"
#include "zarr2/dataset.hpp"
#include "zarr2/fsstore.hpp"

template <typename Store>
inline Observer auto create_superdrops_observer(const Config &config, const Timesteps &tsteps,
                                                Dataset<Store> &dataset) {
  const auto obsstep = (unsigned int)tsteps.get_obsstep();
  const auto maxchunk = int{config.maxchunk};

  CollectDataForDataset<Store> auto sdid = CollectSdId(dataset, maxchunk);
  CollectDataForDataset<Store> auto sdgbxindex = CollectSdgbxindex(dataset, maxchunk);
  CollectDataForDataset<Store> auto xi = CollectXi(dataset, maxchunk);
  CollectDataForDataset<Store> auto radius = CollectRadius(dataset, maxchunk);
  CollectDataForDataset<Store> auto msol = CollectMsol(dataset, maxchunk);
  CollectDataForDataset<Store> auto coord3 = CollectCoord3(dataset, maxchunk);
  CollectDataForDataset<Store> auto coord1 = CollectCoord1(dataset, maxchunk);
  CollectDataForDataset<Store> auto coord2 = CollectCoord2(dataset, maxchunk);

  const auto collect_data =
      coord1 >> coord2 >> coord3 >> msol >> radius >> xi >> sdgbxindex >> sdid;
  return SuperdropsObserver(obsstep, dataset, maxchunk, collect_data);
}

template <typename Store>
inline Observer auto create_gridbox_observer(const Config &config, const Timesteps &tsteps,
                                             Dataset<Store> &dataset) {
  const auto obsstep = (unsigned int)tsteps.get_obsstep();
  const auto maxchunk = int{config.maxchunk};

  const CollectDataForDataset<Store> auto thermo = CollectThermo(dataset, maxchunk, config.ngbxs);
  const CollectDataForDataset<Store> auto windvel = CollectWindVel(dataset, maxchunk, config.ngbxs);
  const CollectDataForDataset<Store> auto nsupers = CollectNsupers(dataset, maxchunk, config.ngbxs);
  const CollectDataForDataset<Store> auto collect_data = nsupers >> windvel >> thermo;
  return WriteToDatasetObserver(obsstep, dataset, collect_data);

  // const Observer auto obst = ThermoObserver(obsstep, dataset, maxchunk, config.ngbxs);
  // const Observer auto obsw = WindVelObserver(obsstep, dataset, maxchunk, config.ngbxs);
  // return obsw >> obst;

  // const Observer auto obsx = StateObserver(obsstep, dataset, maxchunk, config.ngbxs);
  // return obsx;

  // const Observer auto obsn = NsupersObserver(obsstep, dataset, maxchunk, config.ngbxs);
  // return obsn;
}

template <typename Store>
inline Observer auto create_observer2(const Config &config, const Timesteps &tsteps,
                                      Dataset<Store> &dataset) {
  const auto obsstep = (unsigned int)tsteps.get_obsstep();
  const auto maxchunk = int{config.maxchunk};

  const Observer auto obs0 = RunStatsObserver(obsstep, config.stats_filename);
  const Observer auto obs1 = TimeObserver(obsstep, dataset, maxchunk, &step2dimlesstime);
  const Observer auto obs2 = GbxindexObserver(dataset, maxchunk, config.ngbxs);
  const Observer auto obs3 = MassMomentsObserver(obsstep, dataset, maxchunk, config.ngbxs);
  const Observer auto obs4 = MassMomentsRaindropsObserver(obsstep, dataset, maxchunk, config.ngbxs);
  const Observer auto obs6 = TotNsupersObserver(obsstep, dataset, maxchunk);
  const Observer auto obsx = create_gridbox_observer(config, tsteps, dataset);
  const Observer auto obssd = create_superdrops_observer(config, tsteps, dataset);

  return obssd >> obsx >> obs6 >> obs4 >> obs3 >> obs2 >> obs1 >> obs0;
}

/* ---------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------- */

template <typename Store>
inline Observer auto create_observer(const Config &config, const Timesteps &tsteps,
                                     Dataset<Store> &dataset) {
  const auto obsstep = (unsigned int)tsteps.get_obsstep();

  const Observer auto obs0 = StreamOutObserver(obsstep, &step2realtime);

  const Observer auto obs1 = create_observer2(config, tsteps, dataset);

  return obs0 >> obs1;
}

inline InitialConditions auto create_initconds(const Config &config) {
  const InitSupersFromBinary initsupers(config);
  const InitGbxsNull initgbxs(config);

  return InitConds(initsupers, initgbxs);
}

inline CoupledDynamics auto create_coupldyn(const Config &config, const CartesianMaps &gbxmaps,
                                            const unsigned int couplstep,
                                            const unsigned int t_end) {
  const auto h_ndims(gbxmaps.ndims_hostcopy());
  const std::array<size_t, 3> ndims({h_ndims(0), h_ndims(1), h_ndims(2)});

  const auto nsteps = (unsigned int)(std::ceil(t_end / couplstep) + 1);

  return FromFileDynamics(config, couplstep, ndims, nsteps);
}

template <typename Store>
inline auto create_sdm(const Config &config, const Timesteps &tsteps, Dataset<Store> &dataset) {
  const auto couplstep = (unsigned int)tsteps.get_couplstep();
  const GridboxMaps auto gbxmaps =
      create_cartesian_maps(config.ngbxs, config.nspacedims, config.grid_filename);
  const MicrophysicalProcess auto microphys = NullMicrophysicalProcess{};
  const Motion<CartesianMaps> auto movesupers = NullMotion{};
  const Observer auto obs = create_observer(config, tsteps, dataset);
  return SDMMethods(couplstep, gbxmaps, microphys, movesupers, obs);
}

int main(int argc, char *argv[]) {
  Kokkos::Timer kokkostimer;

  /* Read input parameters from configuration file(s) */
  const Config config("/home/m/m300950/CLEO/roughpaper/share/config.txt");
  const Timesteps tsteps(config);  // timesteps for model (e.g. coupling and end time)

  /* Create zarr store for writing output to storage */
  auto store = FSStore(config.zarrbasedir);
  auto dataset = Dataset(store);

  /* Initial conditions for CLEO run */
  const InitialConditions auto initconds = create_initconds(config);

  Kokkos::initialize(argc, argv);
  {
    /* CLEO Super-Droplet Model (excluding coupled dynamics solver) */
    const SDMMethods sdm(create_sdm(config, tsteps, dataset));

    /* Solver of dynamics coupled to CLEO SDM */
    CoupledDynamics auto coupldyn(
        create_coupldyn(config, sdm.gbxmaps, tsteps.get_couplstep(), tsteps.get_t_end()));

    /* coupling between coupldyn and SDM */
    const CouplingComms<FromFileDynamics> auto comms = FromFileComms{};

    /* Run CLEO (SDM coupled to dynamics solver) */
    const RunCLEO runcleo(sdm, coupldyn, comms);
    runcleo(initconds, tsteps.get_t_end());
  }
  Kokkos::finalize();

  const auto ttot = double{kokkostimer.seconds()};
  std::cout << "-------------------------------\n"
               "Total Program Duration: "
            << ttot << "s \n-------------------------------\n";
}
/* ---------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------- */
