/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: main_eurec4a1d.cpp
 * Project: src
 * Created Date: Tuesday 9th April 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Tuesday 18th June 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * runs the CLEO super-droplet model (SDM) for eurec4a 1-D rainshaft example.
 * After make/compiling, execute for example via:
 * ./src/eurec4a1d ../src/config/config.yaml
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "cartesiandomain/add_supers_at_domain_top.hpp"
#include "cartesiandomain/cartesianmaps.hpp"
#include "cartesiandomain/cartesianmotion.hpp"
#include "cartesiandomain/createcartesianmaps.hpp"
#include "cartesiandomain/null_boundary_conditions.hpp"
#include "coupldyn_fromfile/fromfile_cartesian_dynamics.hpp"
#include "coupldyn_fromfile/fromfilecomms.hpp"
#include "gridboxes/gridboxmaps.hpp"
#include "initialise/config.hpp"
#include "initialise/init_all_supers_from_binary.hpp"
#include "initialise/init_supers_from_binary.hpp"
#include "initialise/initgbxsnull.hpp"
#include "initialise/initialconditions.hpp"
#include "initialise/timesteps.hpp"
#include "observers/gbxindex_observer.hpp"
#include "observers/massmoments_observer.hpp"
#include "observers/nsupers_observer.hpp"
#include "observers/observers.hpp"
#include "observers/runstats_observer.hpp"
#include "observers/sdmmonitor/monitor_condensation_observer.hpp"
#include "observers/state_observer.hpp"
#include "observers/streamout_observer.hpp"
#include "observers/superdrops_observer.hpp"
#include "observers/time_observer.hpp"
#include "observers/windvel_observer.hpp"
#include "runcleo/coupleddynamics.hpp"
#include "runcleo/couplingcomms.hpp"
#include "runcleo/runcleo.hpp"
#include "runcleo/sdmmethods.hpp"
#include "superdrops/collisions/breakup.hpp"
#include "superdrops/collisions/breakup_nfrags.hpp"
#include "superdrops/collisions/coalbure.hpp"
#include "superdrops/collisions/coalbure_flag.hpp"
#include "superdrops/collisions/coalescence.hpp"
#include "superdrops/collisions/longhydroprob.hpp"
#include "superdrops/condensation.hpp"
#include "superdrops/microphysicalprocess.hpp"
#include "superdrops/motion.hpp"
#include "superdrops/terminalvelocity.hpp"
#include "zarr/dataset.hpp"
#include "zarr/fsstore.hpp"

// ===================================================
// COUPLED DYNAMICS
// ===================================================

inline CoupledDynamics auto create_coupldyn(const Config &config, const CartesianMaps &gbxmaps,
                                            const unsigned int couplstep,
                                            const unsigned int t_end) {
  const auto h_ndims(gbxmaps.ndims_hostcopy());
  const std::array<size_t, 3> ndims({h_ndims(0), h_ndims(1), h_ndims(2)});

  const auto nsteps = (unsigned int)(std::ceil(t_end / couplstep) + 1);

  return FromFileDynamics(config.get_fromfiledynamics(), couplstep, ndims, nsteps);
}

// ===================================================
// INITIAL CONDITIONS
// ===================================================

inline InitialConditions auto create_initconds(const Config &config) {
  // const InitAllSupersFromBinary initsupers(config.get_initsupersfrombinary());
  const InitSupersFromBinary initsupers(config.get_initsupersfrombinary());
  const InitGbxsNull initgbxs(config.get_ngbxs());

  return InitConds(initsupers, initgbxs);
}

// ===================================================
// GRIDBOXES
// ===================================================

inline GridboxMaps auto create_gbxmaps(const Config &config) {
  const auto gbxmaps = create_cartesian_maps(config.get_ngbxs(), config.get_nspacedims(),
                                             config.get_grid_filename());
  return gbxmaps;
}

// ===================================================
// MOVEMENT
// ===================================================

inline auto create_movement(const Config &config, const Timesteps &tsteps,
                            const CartesianMaps &gbxmaps) {
  const auto terminalv = RogersGKTerminalVelocity{};
  const Motion<CartesianMaps> auto motion =
      CartesianMotion(tsteps.get_motionstep(), &step2dimlesstime, terminalv);

  // const auto boundary_conditions = NullBoundaryConditions{};
  const auto boundary_conditions = AddSupersAtDomainTop(config.get_addsupersatdomaintop());

  return MoveSupersInDomain(gbxmaps, motion, boundary_conditions);
}

// ===================================================
// MICROPHYSICS
// ===================================================

// ------------------------------
// Collision Breakup Rebound and Condensation with nfrags constant from config file
// ------------------------------
inline MicrophysicalProcess auto create_microphysics(const Config &config,
                                                     const Timesteps &tsteps) {
  const auto c_cond = config.get_condensation();
  const MicrophysicalProcess auto cond =
      Condensation(tsteps.get_condstep(), &step2dimlesstime,
                   c_cond.do_alter_thermo, c_cond.maxniters, c_cond.rtol,
                   c_cond.atol, c_cond.MINSUBTSTEP, &realtime2dimless);
  const auto c_breakup = config.get_breakup();
  const PairProbability auto collprob = LongHydroProb();
  const NFragments auto nfrags = ConstNFrags(c_breakup.constnfrags.nfrags);
  const CoalBuReFlag auto coalbure_flag = TSCoalBuReFlag{};
  const MicrophysicalProcess auto coalbure =
      CoalBuRe(tsteps.get_collstep(), &step2realtime, collprob, nfrags, coalbure_flag);
  return cond >> coalbure;
}



// ===================================================
// OBSERVERS
// ===================================================

template <typename Store>
inline Observer auto create_superdrops_observer(const unsigned int interval,
                                                Dataset<Store> &dataset, const int maxchunk) {
  CollectDataForDataset<Store> auto sdid = CollectSdId(dataset, maxchunk);
  CollectDataForDataset<Store> auto sdgbxindex = CollectSdgbxindex(dataset, maxchunk);
  CollectDataForDataset<Store> auto xi = CollectXi(dataset, maxchunk);
  CollectDataForDataset<Store> auto radius = CollectRadius(dataset, maxchunk);
  CollectDataForDataset<Store> auto msol = CollectMsol(dataset, maxchunk);
  CollectDataForDataset<Store> auto coord3 = CollectCoord3(dataset, maxchunk);

  const auto collect_sddata = coord3 >> msol >> radius >> xi >> sdgbxindex >> sdid;
  return SuperdropsObserver(interval, dataset, maxchunk, collect_sddata);
}

template <typename Store>
inline Observer auto create_gridboxes_observer(const unsigned int interval, Dataset<Store> &dataset,
                                               const int maxchunk, const size_t ngbxs) {
  const CollectDataForDataset<Store> auto thermo = CollectThermo(dataset, maxchunk, ngbxs);
  const CollectDataForDataset<Store> auto wvel =
      CollectWindVariable<Store, WvelFunc>(dataset, WvelFunc{}, "wvel", maxchunk, ngbxs);

  const CollectDataForDataset<Store> auto nsupers = CollectNsupers(dataset, maxchunk, ngbxs);

  const CollectDataForDataset<Store> auto collect_gbxdata = nsupers >> wvel >> thermo;
  return WriteToDatasetObserver(interval, dataset, collect_gbxdata);
}

template <typename Store>
inline Observer auto create_observer(const Config &config, const Timesteps &tsteps,
                                     Dataset<Store> &dataset) {
  const auto obsstep = tsteps.get_obsstep();
  const auto maxchunk = config.get_maxchunk();
  const auto ngbxs = config.get_ngbxs();

  const Observer auto obsstats = RunStatsObserver(obsstep, config.get_stats_filename());

  const Observer auto obsstreamout = StreamOutObserver(realtime2step(240), &step2realtime);

  const Observer auto obstime = TimeObserver(obsstep, dataset, maxchunk, &step2dimlesstime);

  const Observer auto obsgindex = GbxindexObserver(dataset, maxchunk, ngbxs);

  const Observer auto obsmm = MassMomentsObserver(obsstep, dataset, maxchunk, ngbxs);

  const Observer auto obsmmrain = MassMomentsRaindropsObserver(obsstep, dataset, maxchunk, ngbxs);

  const Observer auto obsgbx = create_gridboxes_observer(obsstep, dataset, maxchunk, ngbxs);

  const Observer auto obssd = create_superdrops_observer(obsstep, dataset, maxchunk);

  const Observer auto obscond = MonitorCondensationObserver(obsstep, dataset, maxchunk, ngbxs);

  return obscond
        >> obssd
        >> obsgbx
        >> obsmmrain
        >> obsmm
        >> obsgindex
        >> obstime
        >> obsstreamout
        >> obsstats;
}

// ===================================================
// MAIN SUPER DROPLET MODEL
// ===================================================

template <typename Store>
inline auto create_sdm(const Config &config, const Timesteps &tsteps, Dataset<Store> &dataset) {
  const auto couplstep = (unsigned int)tsteps.get_couplstep();
  const GridboxMaps auto gbxmaps(create_gbxmaps(config));
  const MicrophysicalProcess auto microphys(create_microphysics(config, tsteps));
  const MoveSupersInDomain movesupers(create_movement(config, tsteps, gbxmaps));
  const Observer auto obs(create_observer(config, tsteps, dataset));

  return SDMMethods(couplstep, gbxmaps, microphys, movesupers, obs);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    throw std::invalid_argument("configuration file(s) not specified");
  }

  Kokkos::Timer kokkostimer;

  /* Read input parameters from configuration file(s) */
  const std::filesystem::path config_filename(argv[1]);  // path to configuration file
  const Config config(config_filename);
  const Timesteps tsteps(config.get_timesteps());

  /* Create Xarray dataset wit Zarr backend for writing output data to a store */
  auto store = FSStore(config.get_zarrbasedir());
  auto dataset = Dataset(store);

  /* Initial conditions for CLEO run */
  const InitialConditions auto initconds = create_initconds(config);

  /* Initialise Kokkos parallel environment */
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
  std::cout << "-----\n Total Program Duration: " << ttot << "s \n-----\n";

  return 0;
}
