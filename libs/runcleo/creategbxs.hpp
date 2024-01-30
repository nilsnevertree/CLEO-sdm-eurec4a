/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: creategbxs.hpp
 * Project: runcleo
 * Created Date: Wednesday 18th October 2023
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
 * file for structure to create a dualview of
 * gridboxes from using some initial conditions
 */

#ifndef LIBS_RUNCLEO_CREATEGBXS_HPP_
#define LIBS_RUNCLEO_CREATEGBXS_HPP_

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Pair.hpp>

#include "../kokkosaliases.hpp"
#include "gridboxes/findrefs.hpp"
#include "gridboxes/gbxindex.hpp"
#include "gridboxes/gridbox.hpp"
#include "gridboxes/gridboxmaps.hpp"
#include "gridboxes/supersingbx.hpp"
#include "superdrops/state.hpp"
#include "superdrops/superdrop.hpp"

template <GridboxMaps GbxMaps, typename GbxInitConds>
dualview_gbx create_gbxs(const GbxMaps &gbxmaps, const GbxInitConds &gbxic,
                         const viewd_supers totsupers);

class GenGridbox {
 private:
  std::shared_ptr<Gbxindex::Gen> GbxindexGen;  // pointer to gridbox index generator
  std::vector<double> presss;
  std::vector<double> temps;
  std::vector<double> qvaps;
  std::vector<double> qconds;
  std::vector<std::pair<double, double>> wvels;
  std::vector<std::pair<double, double>> uvels;
  std::vector<std::pair<double, double>> vvels;

  State state_at(const unsigned int ii, const double volume) const;

 public:
  template <typename GbxInitConds>
  explicit GenGridbox(const GbxInitConds &gbxic)
      : GbxindexGen(std::make_shared<Gbxindex::Gen>()),
        presss(gbxic.press()),
        temps(gbxic.temp()),
        qvaps(gbxic.qvap()),
        qconds(gbxic.qcond()),
        wvels(gbxic.wvel()),
        uvels(gbxic.uvel()),
        vvels(gbxic.vvel()) {}

  template <GridboxMaps GbxMaps>
  Gridbox operator()(const unsigned int ii, const GbxMaps &gbxmaps,
                     const viewd_supers totsupers) const {
    const auto gbxindex = GbxindexGen->next(ii);
    const auto volume = gbxmaps.get_gbxvolume(gbxindex.value);
    const State state(state_at(ii, volume));

    return Gridbox(gbxindex, state, totsupers);
  }

  template <GridboxMaps GbxMaps>
  Gridbox operator()(const HostTeamMember &team_member, const unsigned int ii,
                     const GbxMaps &gbxmaps, const viewd_supers totsupers,
                     const viewd_constsupers::HostMirror h_totsupers) const {
    const auto gbxindex = GbxindexGen->next(ii);
    const auto volume = gbxmaps.get_gbxvolume(gbxindex.value);
    const State state(state_at(ii, volume));
    const kkpair_size_t refs(find_refs(team_member, h_totsupers, gbxindex.value));

    return Gridbox(gbxindex, state, totsupers, refs);
  }
};

/* initialise the host view of gridboxes
using some data from a GbxInitConds instance
e.g. for each gridbox's volume */
template <GridboxMaps GbxMaps>
inline void initialise_gbxs_on_host(const GbxMaps &gbxmaps, const GenGridbox &gen,
                                    const viewd_supers totsupers, const viewh_gbx h_gbxs);

/* initialise a dualview of gridboxes (on host and device
memory) using data from a GbxInitConds instance to initialise
the host view and then syncing the view to the device */
template <GridboxMaps GbxMaps, typename GbxInitConds>
inline dualview_gbx initialise_gbxs(const GbxMaps &gbxmaps, const GbxInitConds &gbxic,
                                    const viewd_supers totsupers);

void is_gbxinit_complete(const size_t ngbxs_from_maps, dualview_gbx gbxs);

/* print gridboxes information */
void print_gbxs(const viewh_constgbx gbxs);

template <GridboxMaps GbxMaps, typename GbxInitConds>
dualview_gbx create_gbxs(const GbxMaps &gbxmaps, const GbxInitConds &gbxic,
                         const viewd_supers totsupers) {
  std::cout << "\n--- create gridboxes ---\n"
            << "initialising\n";
  const dualview_gbx gbxs(initialise_gbxs(gbxmaps, gbxic, totsupers));

  std::cout << "checking initialisation\n";
  is_gbxinit_complete(gbxmaps.maps_size() - 1, gbxs);
  print_gbxs(gbxs.view_host());

  std::cout << "--- create gridboxes: success ---\n";

  return gbxs;
}

/* initialise a view of superdrops (on device memory)
using data from an InitData instance for their initial
gbxindex, spatial coordinates and attributes */
template <GridboxMaps GbxMaps, typename GbxInitConds>
inline dualview_gbx initialise_gbxs(const GbxMaps &gbxmaps, const GbxInitConds &gbxic,
                                    const viewd_supers totsupers) {
  // create dualview for gridboxes on device and host memory
  dualview_gbx gbxs("gbxs", gbxic.get_ngbxs());

  // initialise gridboxes on host
  const GenGridbox gen(gbxic);
  gbxs.sync_host();
  initialise_gbxs_on_host(gbxmaps, gen, totsupers, gbxs.view_host());
  gbxs.modify_host();

  // update device gridbox view to match host's gridbox view
  gbxs.sync_device();

  return gbxs;
}

/* initialise the host (!) view of gridboxes
using some data from gridbox generator 'gen'
e.g. for each gridbox's volume.
Kokkos::parallel_for([...]) is equivalent to:
for (size_t ii(0); ii < ngbxs; ++ii) {[...]}
when in serial */
template <GridboxMaps GbxMaps>
inline void initialise_gbxs_on_host(const GbxMaps &gbxmaps, const GenGridbox &gen,
                                    const viewd_supers totsupers, const viewh_gbx h_gbxs) {
  const size_t ngbxs(h_gbxs.extent(0));

  /* equivalent serial version of parallel_for loop below
  for (size_t ii(0); ii < ngbxs; ++ii)
  {
    h_gbxs(ii) = gen(ii, gbxmaps, totsupers);
  }
  */

  auto h_totsupers =
      Kokkos::create_mirror_view(totsupers);  // mirror totsupers in case view is on device memory
  Kokkos::deep_copy(h_totsupers, totsupers);

  Kokkos::parallel_for("initialise_gbxs_on_host", HostTeamPolicy(ngbxs, Kokkos::AUTO()),
                       [=](const HostTeamMember &team_member) {
                         const int ii = team_member.league_rank();

                         const Gridbox gbx(gen(team_member, ii, gbxmaps, totsupers, h_totsupers));

                         /* use 1 thread on host to write gbx to view */
                         team_member.team_barrier();
                         if (team_member.team_rank() == 0) {
                           h_gbxs(ii) = gbx;
                         }
                       });
}

#endif  // LIBS_RUNCLEO_CREATEGBXS_HPP_
