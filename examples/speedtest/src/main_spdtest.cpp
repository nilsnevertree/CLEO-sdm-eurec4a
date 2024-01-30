/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: main_spdtest.cpp
 * Project: src
 * Created Date: Thursday 12th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Wednesday 17th January 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * runs the CLEO super-droplet model (SDM)
 * after make/compiling, execute for example via:
 * ./src/spdtest ../src/config/config.txt
 */

#include <concepts>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string_view>

#include <Kokkos_Core.hpp>

#include "cartesiandomain/cartesianmaps.hpp"
#include "cartesiandomain/cartesianmotion.hpp"
#include "cartesiandomain/createcartesianmaps.hpp"
#include "coupldyn_fromfile/fromfile_cartesian_dynamics.hpp"
#include "coupldyn_fromfile/fromfilecomms.hpp"
#include "gridboxes/gridboxmaps.hpp"
#include "initialise/config.hpp"
#include "initialise/initgbxs_null.hpp"
#include "initialise/initsupers_frombinary.hpp"
#include "initialise/timesteps.hpp"
#include "observers/gbxindexobs.hpp"
#include "observers/massmomentsobs.hpp"
#include "observers/nsupersobs.hpp"
#include "observers/observers.hpp"
#include "observers/printobs.hpp"
#include "observers/runstats.hpp"
#include "observers/stateobs.hpp"
#include "observers/supersattrsobs.hpp"
#include "observers/timeobs.hpp"
#include "runcleo/coupleddynamics.hpp"
#include "runcleo/couplingcomms.hpp"
#include "runcleo/initialconditions.hpp"
#include "runcleo/runcleo.hpp"
#include "runcleo/sdmmethods.hpp"
#include "superdrops/coalescence.hpp"
#include "superdrops/collisionprobs/longhydroprob.hpp"
#include "superdrops/condensation.hpp"
#include "superdrops/microphysicalprocess.hpp"
#include "superdrops/motion.hpp"
#include "superdrops/terminalvelocity.hpp"
#include "zarr/fsstore.hpp"
#include "zarr/superdropattrsbuffers.hpp"
#include "zarr/superdropsbuffers.hpp"

inline CoupledDynamics auto create_coupldyn(const Config &config, const CartesianMaps &gbxmaps,
                                            const unsigned int couplstep,
                                            const unsigned int t_end) {
  const auto h_ndims(gbxmaps.ndims_hostcopy());
  const std::array<size_t, 3> ndims({h_ndims(0), h_ndims(1), h_ndims(2)});

  const auto nsteps = (unsigned int)(std::ceil(t_end / couplstep) + 1);

  return FromFileDynamics(config, couplstep, ndims, nsteps);
}

inline InitialConditions auto create_initconds(const Config &config) {
  const InitSupersFromBinary initsupers(config);
  const InitGbxsNull initgbxs(config);

  return InitConds(initsupers, initgbxs);
}

inline GridboxMaps auto create_gbxmaps(const Config &config) {
  const auto gbxmaps = create_cartesian_maps(config.ngbxs, config.nspacedims, config.grid_filename);
  return gbxmaps;
}

inline MicrophysicalProcess auto config_condensation(const Config &config,
                                                     const Timesteps &tsteps) {
  return Condensation(tsteps.get_condstep(), config.doAlterThermo, config.cond_iters,
                      &step2dimlesstime, config.cond_rtol, config.cond_atol, config.cond_SUBTSTEP,
                      &realtime2dimless);
}

inline MicrophysicalProcess auto config_collisions(const Config &config, const Timesteps &tsteps) {
  const PairProbability auto coalprob = LongHydroProb(1.0);
  const MicrophysicalProcess auto coal = CollCoal(tsteps.get_collstep(), &step2realtime, coalprob);

  return coal;
}

inline MicrophysicalProcess auto create_microphysics(const Config &config,
                                                     const Timesteps &tsteps) {
  const MicrophysicalProcess auto cond = config_condensation(config, tsteps);

  const MicrophysicalProcess auto colls = config_collisions(config, tsteps);

  return colls >> cond;
}

inline Motion<CartesianMaps> auto create_motion(const unsigned int motionstep) {
  const auto terminalv = RogersGKTerminalVelocity{};

  return CartesianMotion(motionstep, &step2dimlesstime, terminalv);
}

inline Observer auto create_supersattrs_observer(const unsigned int interval, FSStore &store,
                                                 const int maxchunk) {
  SuperdropsBuffers auto buffers = SdIdBuffer() >> XiBuffer() >> MsolBuffer() >> RadiusBuffer() >>
                                   Coord3Buffer() >> Coord1Buffer() >> Coord2Buffer() >>
                                   SdgbxindexBuffer();
  return SupersAttrsObserver(interval, store, maxchunk, buffers);
}

inline Observer auto create_bulk_observer(const Config &config, const Timesteps &tsteps,
                                          FSStore &store) {
  const auto obsstep = (unsigned int)tsteps.get_obsstep();
  const auto maxchunk = int{config.maxchunk};

  const Observer auto obs1 = PrintObserver(obsstep, &step2realtime);

  const Observer auto obs2 = TimeObserver(obsstep, store, maxchunk, &step2dimlesstime);

  const Observer auto obs3 = TotNsupersObserver(obsstep, store, maxchunk);

  const Observer auto obs4 = GbxindexObserver(store, maxchunk);

  const Observer auto obs5 = StateObserver(obsstep, store, maxchunk, config.ngbxs);

  const Observer auto obs6 = NsupersObserver(obsstep, store, maxchunk, config.ngbxs);

  const Observer auto obs7 = MassMomentsObserver(obsstep, store, maxchunk, config.ngbxs);

  return obs1 >> obs2 >> obs3 >> obs4 >> obs5 >> obs6 >> obs7;
}

inline Observer auto create_superdrops_observer(const Config &config, const Timesteps &tsteps,
                                                FSStore &store) {
  const auto obsstep = (unsigned int)tsteps.get_obsstep();
  const auto maxchunk = int{config.maxchunk};

  const Observer auto obs8 = NrainsupersObserver(obsstep, store, maxchunk, config.ngbxs);

  const Observer auto obs9 = RainMassMomentsObserver(obsstep, store, maxchunk, config.ngbxs);

  const Observer auto obs10 = create_supersattrs_observer(obsstep, store, maxchunk);

  return obs8 >> obs9 >> obs10;
}

inline Observer auto create_observer(const Config &config, const Timesteps &tsteps,
                                     FSStore &store) {
  const Observer auto obs0 = RunStatsObserver(tsteps.get_obsstep(), config.stats_filename);

  const Observer auto obs_blk = create_bulk_observer(config, tsteps, store);

  const Observer auto obs_sd = create_superdrops_observer(config, tsteps, store);

  return obs0 >> obs_blk >> obs_sd;
}

inline auto create_sdm(const Config &config, const Timesteps &tsteps, FSStore &store) {
  const auto couplstep = (unsigned int)tsteps.get_couplstep();
  const GridboxMaps auto gbxmaps(create_gbxmaps(config));
  const MicrophysicalProcess auto microphys(create_microphysics(config, tsteps));
  const Motion<CartesianMaps> auto movesupers(create_motion(tsteps.get_motionstep()));
  const Observer auto obs(create_observer(config, tsteps, store));

  return SDMMethods(couplstep, gbxmaps, microphys, movesupers, obs);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    throw std::invalid_argument("configuration file(s) not specified");
  }

  Kokkos::Timer kokkostimer;

  /* Read input parameters from configuration file(s) */
  const std::string_view config_filename(argv[1]);  // path to configuration file
  const Config config(config_filename);
  const Timesteps tsteps(config);  // timesteps for model (e.g. coupling and end time)

  /* Create zarr store for writing output to storage */
  FSStore fsstore(config.zarrbasedir);

  /* Initial conditions for CLEO run */
  const InitialConditions auto initconds = create_initconds(config);

  /* Initialise Kokkos parallel environment */
  Kokkos::initialize(argc, argv);
  {
    /* CLEO Super-Droplet Model (excluding coupled dynamics solver) */
    const SDMMethods sdm(create_sdm(config, tsteps, fsstore));

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
