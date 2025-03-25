/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: main_eurec4a1d.cpp
 * Project: src
 * Created Date: Tuesday 9th April 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors: Nils niebaum (NN)
 * -----
 * Last Modified: Friday 27th January 2025
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

#include "zarr/dataset.hpp"
#include "cartesiandomain/cartesianmaps.hpp"
#include "cartesiandomain/createcartesianmaps.hpp"
#include "cartesiandomain/movement/add_supers_at_domain_top.hpp"
#include "cartesiandomain/movement/cartesian_motion.hpp"
#include "cartesiandomain/movement/cartesian_movement.hpp"
#include "coupldyn_fromfile/fromfile_cartesian_dynamics.hpp"
#include "coupldyn_fromfile/fromfilecomms.hpp"
#include "gridboxes/boundary_conditions.hpp"
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
#include "observers/sdmmonitor/monitor_condensation_observer.hpp"
#include "observers/state_observer.hpp"
#include "observers/streamout_observer.hpp"
#include "observers/superdrops_observer.hpp"
#include "observers/time_observer.hpp"
#include "observers/thermo_observer.hpp"
#include "observers/totnsupers_observer.hpp"
#include "observers/windvel_observer.hpp"
#include "runcleo/coupleddynamics.hpp"
#include "runcleo/couplingcomms.hpp"
#include "runcleo/runcleo.hpp"
#include "runcleo/sdmmethods.hpp"
#include "superdrops/collisions/coalescence.hpp"
#include "superdrops/collisions/longhydroprob.hpp"
#include "superdrops/condensation.hpp"
#include "superdrops/microphysicalprocess.hpp"
#include "superdrops/motion.hpp"
#include "superdrops/terminalvelocity.hpp"
#include "zarr/fsstore.hpp"

// ===================================================
// COUPLED DYNAMICS
// ===================================================

inline CoupledDynamics auto create_coupldyn(const Config &config, const CartesianMaps &gbxmaps,
                                            const unsigned int couplstep,
                                            const unsigned int t_end) {
  const auto h_ndims = gbxmaps.get_global_ndims_hostcopy();
  const std::array<size_t, 3> ndims({h_ndims(0), h_ndims(1), h_ndims(2)});

  const auto nsteps = (unsigned int)(std::ceil(t_end / couplstep) + 1);

  return FromFileDynamics(config.get_fromfiledynamics(), couplstep, ndims, nsteps);
}

// ===================================================
// INITIAL CONDITIONS
// ===================================================

template <GridboxMaps GbxMaps>
inline InitialConditions auto create_initconds(const Config &config, const GbxMaps &gbxmaps) {
  const auto initsupers = InitSupersFromBinary(config.get_initsupersfrombinary(), gbxmaps);
  const auto initgbxs = InitGbxsNull(gbxmaps.get_local_ngridboxes_hostcopy());

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

inline auto create_movement(const Config &config,
                            const unsigned int motionstep,
                            const CartesianMaps &gbxmaps) {
  const auto terminalv = RogersGKTerminalVelocity{};
  const Motion<CartesianMaps> auto motion =
      CartesianMotion(tsteps.get_motionstep(), &step2dimlesstime, terminalv);

  // const BoundaryConditions<CartesianMaps> auto boundary_conditions = NullBoundaryConditions{};
  const BoundaryConditions<CartesianMaps> auto boundary_conditions =
      AddSupersAtDomainTop(config.get_addsupersatdomaintop());

  return cartesian_movement(gbxmaps, motion, boundary_conditions);
}
// ===================================================
// MICROPHYSICS
// ===================================================

// ------------------------------
// condensation only
// ------------------------------
inline MicrophysicalProcess auto create_microphysics(const Config &config,
                                                     const Timesteps &tsteps) {
  const auto c = config.get_condensation();
  const MicrophysicalProcess auto cond =
      Condensation(tsteps.get_condstep(), &step2dimlesstime, c.do_alter_thermo, c.maxniters, c.rtol,
                   c.atol, c.MINSUBTSTEP, &realtime2dimless);
  return cond;
};
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

  // const Observer auto obsstats = RunStatsObserver(obsstep, config.get_stats_filename());

  const Observer auto obsstreamout = StreamOutObserver(realtime2step(240), &step2realtime);

  const Observer auto obstime = TimeObserver(obsstep, dataset, maxchunk, &step2dimlesstime);

  const Observer auto obsgindex = GbxindexObserver(dataset, maxchunk, ngbxs);

  const Observer auto obsnsupers = NsupersObserver(obsstep, dataset, maxchunk, ngbxs);

  const Observer auto obsmm = MassMomentsObserver(obsstep, dataset, maxchunk, ngbxs);

  const Observer auto obsmmrain = MassMomentsRaindropsObserver(obsstep, dataset, maxchunk, ngbxs);

  const Observer auto obsgbx = create_gridboxes_observer(obsstep, dataset, maxchunk, ngbxs);

  const Observer auto obssd = create_superdrops_observer(obsstep, dataset, maxchunk);

  const Observer auto obscond = MonitorCondensationObserver(obsstep, dataset, maxchunk, ngbxs);

  return obscond
        // >> obsstats
        >> obsstreamout
        >> obstime
        >> obsgindex
        >> obsnsupers
        >> obsmm
        >> obsmmrain
        >> obscond
        >> obsgbx
        >> obssd;
}

// ===================================================
// MAIN SUPER DROPLET MODEL
// ===================================================

template <typename Store>
inline auto create_sdm(const Config &config, const Timesteps &tsteps, Dataset<Store> &dataset) {
  const auto couplstep = (unsigned int)tsteps.get_couplstep();
  const GridboxMaps auto gbxmaps(create_gbxmaps(config));
  const MicrophysicalProcess auto microphys(create_microphysics(config, tsteps));
  const MoveSupersInDomain movesupers(create_movement(config, tsteps.get_motionstep(), gbxmaps));
  const Observer auto obs(create_observer(config, tsteps, dataset));

  return SDMMethods(couplstep, gbxmaps, microphys, movesupers, obs);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    throw std::invalid_argument("configuration file(s) not specified");
  }

  MPI_Init(&argc, &argv);

  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if (comm_size > 1) {
    std::cout << "ERROR: The current example is not prepared"
              << " to be run with more than one MPI process" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Kokkos::Timer kokkostimer;

  /* Read input parameters from configuration file(s) */
  const std::filesystem::path config_filename(argv[1]);  // path to configuration file
  const Config config(config_filename);

  /* Initialise Kokkos parallel environment */
  Kokkos::initialize(config.get_kokkos_initialization_settings());
  {
    Kokkos::print_configuration(std::cout);

    /* Create timestepping parameters from configuration */
    const Timesteps tsteps(config.get_timesteps());

    /* Create Xarray dataset wit Zarr backend for writing output data to a store */
    auto store = FSStore(config.get_zarrbasedir());
    auto dataset = Dataset(store);

    /* CLEO Super-Droplet Model (excluding coupled dynamics solver) */
    const SDMMethods sdm(create_sdm(config, tsteps, dataset));

    /* Solver of dynamics coupled to CLEO SDM */
    CoupledDynamics auto coupldyn(
        create_coupldyn(config, sdm.gbxmaps, tsteps.get_couplstep(), tsteps.get_t_end()));
    /* coupling between coupldyn and SDM */
      const CouplingComms<CartesianMaps, FromFileDynamics> auto comms = FromFileComms{};

    /* Initial conditions for CLEO run */
    const InitialConditions auto initconds = create_initconds(config, sdm.gbxmaps);

    /* Run CLEO (SDM coupled to dynamics solver) */
    const RunCLEO runcleo(sdm, coupldyn, comms);
    runcleo(initconds, tsteps.get_t_end());
  }
  Kokkos::finalize();

  const auto ttot = double{kokkostimer.seconds()};
  std::cout << "-----\n Total Program Duration: " << ttot << "s \n-----\n";

  MPI_Finalize();

  return 0;
}
