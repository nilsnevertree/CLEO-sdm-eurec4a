// Author: Clara Bayley
// File: timesteps.cpp
/* Structs for handling values of
timstep variables for SDM */

#include "timesteps.hpp"


Timesteps::Timesteps(const double CONDTSTEP, const double COLLTSTEP,
const double MOTIONTSTEP, const double COUPLTSTEP, const double T_END)
/* (dimensionless) double's that are timesteps in config struct
are converted into integer values of model timesteps using
model_step and secd template functions created using std::chrono library.
Throw error if after convertion into model timestep, any
timestep = 0 */
    : condsubstep(realtime2timestep(CONDTSTEP)),
      collsubstep(realtime2timestep(COLLTSTEP)),
      motionstep(realtime2timestep(MOTIONTSTEP)),
      couplstep(realtime2timestep(COUPLTSTEP)),
      t_end(realtime2timestep(T_END))
{
  if ((condsubstep == 0) | (collsubstep == 0) | (motionstep == 0) |
      (couplstep == 0) | (t_end == 0))
  {
    const std::string err("A model step = 0, possibly due to bad conversion"
                          " of a real timestep [s]. Consider increasing X in"
                          " std::ratio<1, X> used for definition of model_step");
    throw std::invalid_argument(err);
  }

  if ((couplstep < condsubstep) |
      (couplstep < collsubstep))
  {
    const std::string err("invalid model steps: an sdm substep is larger"
                          " than the coupling step");
    throw std::invalid_argument(err);
  }

  if ((outstep < condstep) |
      (outstep < collstep) |
      (outstep < sedistep))
  {
    throw std::invalid_argument("ERROR! OUTSTEP MODEL TIMESTEP less than cond"
                                "coll or sedi tstep. undefined sdm timstepping");
  }

}