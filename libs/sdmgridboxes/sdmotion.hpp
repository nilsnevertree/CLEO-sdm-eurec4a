// Author: Clara Bayley
// File: "sdmotion.hpp"
/* Header file for functions related to
updatings superdroplets positions 
(updating their
coordinates according to equations of motion) */

#ifndef SDMOTION_HPP
#define SDMOTION_HPP

#include <concepts>
#include <limits>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <numbers>
#include <utility>
#include <functional>

#include "superdrop_solver/superdrop.hpp"
#include "superdrop_solver/terminalvelocity.hpp"
#include "superdrop_solver/thermostate.hpp"
#include "./gridbox.hpp"
#include "./maps4gridboxes.hpp"

bool cfl_criteria(const Maps4GridBoxes &gbxmaps,
                  const unsigned int gbxindex,
                  const double delta3, const double delta1,
                  const double delta2);
/* returns false if any of z, x or y (3,1,2) directions
  do not meet their cfl criterion. For each direction,
  Criterion is C = delta[X] / gridstep =< 1 where the
  gridstep is calculated from the gridbox boundaries map */

inline bool cfl_criterion(const double gridstep,
                          const double sdstep)
/* sdstep = change in superdroplet coordinate position.
returns false if cfl criterion, C = sdstep / gridstep, > 1 */
{
  return (sdstep <= gridstep);
}

template <typename M>
concept SdMotion = requires(M m, const int currenttimestep,
                            const GridBox &gbx,
                            const Maps4GridBoxes &gbxmaps,
                            Superdrop &superdrop)
/* concept SdMotion is all types that meet requirements
(constraints) of void function called "move_superdroplet"
which takes a ThermoState and Superdrop as arguments */
{
  {
    m.next_move(currenttimestep)
    } -> std::convertible_to<int>;
  {
    m.on_move(currenttimestep)
    } -> std::convertible_to<bool>;
  {
    m.change_superdroplet_coords(gbxmaps, gbx, superdrop)
  };
};

struct NullMotion
{
  int next_move(const int currenttimestep) const
  {
    return std::numeric_limits<int>::max();
  }

  bool on_move(const int currenttimestep) const
  {
    return false;
  }

  void change_superdroplet_coords(const Maps4GridBoxes &gbxmaps,
                                  const GridBox &gbx,
                                  Superdrop &superdrop) const {}
};

template <VelocityFormula TerminalVelocity>
class NoInterpMoveWithSedimentation
{
private:
  const int interval;                 // integer timestep for movement
  const double delt;                  // equivalent of interval as dimensionless time
  
  TerminalVelocity terminalv; // returns terminal velocity given a superdroplet

  double deltacoord(const double vel) const
  /* returns change in a coord given a velocity component 'vel' */
  {
    return vel * delt;
  }

public:
  NoInterpMoveWithSedimentation(const int interval,
                        const std::function<double(int)> int2time,
                        const TerminalVelocity terminalv)
      : interval(interval),
        delt(int2time(interval)),
        terminalv(terminalv) {}

  int next_move(const int t) const
  {
    return ((t / interval) + 1) * interval;
  }

  bool on_move(const int t) const
  {
    return t % interval == 0;
  }

  void change_superdroplet_coords(const Maps4GridBoxes &gbxmaps,
                                  const GridBox &gbx,
                                  Superdrop &drop) const
  /* very crude method to forward timestep the velocity
  using the velocity from the gridbox thermostate, ie.
  without interpolation to the SD position and using
  single step forward euler method to integrate dx/dt */
  {
  const double delta3 = deltacoord(gbx.state.wvel - terminalv(drop)); // w wind + terminal velocity
  const double delta1 = deltacoord(gbx.state.uvel); // u component of wind velocity
  const double delta2 = deltacoord(gbx.state.vvel); // v component of wind velocity (y=2)

  cfl_criteria(gbxmaps, gbx.gbxindex, delta3, delta1, delta2);

  drop.coord3 += delta3;
  drop.coord1 += delta1;
  drop.coord2 += delta2;
  }
};

class Prescribed2DFlow
/* Fixed 2D flow with constant density from
Arabas et al. 2015 with lengthscales
xlength = 2*pi*xtilda and zlength = pi*ztilda */
{
private:
  const double ztilda;
  const double xtilda;
  const double wamp;
  const std::function<double(ThermoState)> rhotilda; // function for normalised rho(z)

public:
  Prescribed2DFlow(const double zlength,
                   const double xlength,
                   const double wmax,
                   const std::function<double(ThermoState)> rhotilda);

  double prescribed_wvel(const ThermoState &state, const double zcoord,
                         const double xcoord) const;

  double prescribed_uvel(const ThermoState &state, const double zcoord,
                         const double xcoord) const;
};

class MoveWith2DFixedFlow
{
private:
  const int interval;                 // integer timestep for movement
  const double delt;                  // equivalent of interval as dimensionless time

  const Prescribed2DFlow flow2d; // method to get wvel and uvel from 2D flow field

  std::pair<double, double> predictor_corrector(const ThermoState &state,
                                                const double coord3,
                                                const double coord1) const;
  /* returns change in (z,x) coordinates = (delta3, delta1)
  obtained using predictor-corrector method and velocities
  calculated from a Prescribed2DFlow */

public:
  MoveWith2DFixedFlow(const int interval,
                      const std::function<double(int)> int2time,
                      const Prescribed2DFlow flow2d)
      : interval(interval),
        delt(int2time(interval)),
        flow2d(flow2d) {}

  MoveWith2DFixedFlow(const int interval,
                      const std::function<double(int)> int2time,
                      const double zlength,
                      const double xlength,
                      const double wmax,
                      const std::function<double(ThermoState)> rhotilda)
      : interval(interval),
        delt(int2time(interval)),
        flow2d(Prescribed2DFlow(zlength, xlength, wmax, rhotilda)) {}

  int next_move(const int t) const
  {
    return ((t / interval) + 1) * interval;
  }

  bool on_move(const int t) const
  {
    return t % interval == 0;
  }

  void change_superdroplet_coords(const Maps4GridBoxes &gbxmaps,
                                  const GridBox &gbx,
                                  Superdrop &drop) const;
  /* Use predictor-corrector scheme from Grabowksi et al. 2018
  (similar to Arabas et al. 2015) to update a SD position.
  The velocity required for this scheme is determined
  from the PrescribedFlow2D instance */
};

#endif // SDMOTION_HPP