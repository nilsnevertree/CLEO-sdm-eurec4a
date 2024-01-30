/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: thermodynamic_equations.hpp
 * Project: superdrops
 * Created Date: Wednesday 25th October 2023
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
 * Header file for functions that return
 * Left Hand Side of thermodynamic equations.
 * Equations referenced as (eqn [X.YY])
 * are from "An Introduction To Clouds From The
 * Microscale to Climate" by Lohmann, Luond
 * and Mahrt, 1st edition.
 * */

#ifndef LIBS_SUPERDROPS_THERMODYNAMIC_EQUATIONS_HPP_
#define LIBS_SUPERDROPS_THERMODYNAMIC_EQUATIONS_HPP_

#include <cassert>

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

#include "../cleoconstants.hpp"
#include "./superdrop.hpp"

namespace dlc = dimless_constants;
namespace DC = dimmed_constants;

/* (dimensionless) specific heat capacity of a moist parcel of air */
KOKKOS_INLINE_FUNCTION
double moist_specifc_heat(const double qvap, const double qcond) {
  return dlc::Cp_dry + dlc::Cp_v * (qvap) + dlc::C_l * (qcond);
}

/* calculate the superaturation ratio given the saturaion
pressure, psat, the ambient pressure, press, and the vapour
mass mixing ratio, qvap. supersaturation ratio = 's_ratio',
s_ratio = p_vapour/psat (ie. relative humidity) */
KOKKOS_INLINE_FUNCTION
double supersaturation_ratio(const double press, const double qvap, const double psat) {
  return (press * qvap) / ((dlc::Mr_ratio + qvap) * psat);
}

/* Calculates a and b factors for kohler curve.
Calculates value of a in raoult factor (exp^(a/r))
to account for effect of dissolved solute
on radial growth of droplet. Calculates value of b
in kelvin factor (1-b/r^3) to account for curvature on
radial growth of droplet. Using equations from
"An Introduction To Clouds...." (see note at top of file) */
KOKKOS_INLINE_FUNCTION
Kokkos::pair<double, double> kohler_factors(const Superdrop &drop, const double temp) {
  constexpr double akoh_constant = 3.3e-7 / (dlc::TEMP0 * dlc::R0);
  const auto akoh = akoh_constant / temp;  // dimensionless version of eqn [6.24]

  constexpr double bkoh_constant = 4.3e-6 * dlc::RHO0 / dlc::MR0;
  const auto msol = drop.get_msol();
  const auto ionic = drop.get_ionic();
  const auto mr_sol = drop.get_mr_sol();
  const auto bkoh = bkoh_constant * msol * ionic / mr_sol;  // dimensionless version of eqn [6.22]

  return {akoh, bkoh};  // {a, b} = {raoult, kelvin} kohler factors
}

/* Calculate the equilibrium vapor pressure of water over liquid
water ie. the saturation pressure (psat). Equation taken from
Bjorn Steven's "make_tetens" python function from his module
"moist_thermodynamics.saturation_vapour_pressures" available
on gitlab. Original paper "Murray, F. W. On the Computation of
Saturation Vapor Pressure. Journal of Applied Meteorology
and Climatology 6, 203–204 (1967)." Note function is called
with conversion to real temp /K = T*Temp0 and from real psat
to dimensionless psat = psat/P0. */
KOKKOS_FUNCTION
double saturation_pressure(const double temp);

/* Calculate the equilibrium vapor pressure of water over
liquid water ie. the saturation pressure (psat). Equation taken from
python module typhon.physics.thermodynamics.e_eq_water_mk
with conversion to real temp /K = T*Temp0 and from
real psat to dimensionless psat = psat/P0. */
KOKKOS_FUNCTION
double saturation_pressure_murphy_koop(const double temp);

/* Calculate dimensionless Fkl and Fdl heat and vapour
diffusion factors in equation for radial growth of droplets
according to equations from "An Introduction To Clouds...."
(see note at top of file). fkl is first item of returned
pair, fdl is second. */
KOKKOS_FUNCTION
double diffusion_factor(const double press, const double temp, const double psat);

/* -----  ----- TODO: move functions below to .cpp file ----- ----- */

/* Calculate the equilibrium vapor pressure of water over liquid
water ie. the saturation pressure (psat). Equation taken from
Bjorn Steven's "make_tetens" python function from his module
"moist_thermodynamics.saturation_vapour_pressures" available
on gitlab. Original paper "Murray, F. W. On the Computation of
Saturation Vapor Pressure. Journal of Applied Meteorology
and Climatology 6, 203–204 (1967)." Note function is called
with conversion to real temp /K = T*Temp0 and from real psat
to dimensionless psat = psat/P0. */
KOKKOS_FUNCTION
double saturation_pressure(const double temp) {
  assert((temp > 0) && "psat ERROR: temperature must be larger than 0K.");

  constexpr double A = 17.4146;     // constants from Bjorn Gitlab originally from paper
  constexpr double B = 33.639;      // ditto
  constexpr double TREF = 273.16;   // Triple point temperature [K] of water
  constexpr double PREF = 611.655;  // Triple point pressure [Pa] of water

  const auto T = double{temp * dlc::TEMP0};  // real T [K]

  return (PREF * Kokkos::exp(A * (T - TREF) / (T - B))) / dlc::P0;  // dimensionless psat
}

/* Calculate the equilibrium vapor pressure of water over
liquid water ie. the saturation pressure (psat). Equation taken
from python module typhon.physics.thermodynamics.e_eq_water_mk
with conversion to real temp /K = T*Temp0 and from
real psat to dimensionless psat = psat/P0. */
KOKKOS_FUNCTION
double saturation_pressure_murphy_koop(const double temp) {
  assert((temp > 0) && "psat ERROR: temperature must be larger than 0K.");

  const auto T = double{temp * dlc::TEMP0};  // real T [K]

  const auto lnpsat =
      double{54.842763  // ln(psat) [Pa]
             - 6763.22 / T - 4.21 * log(T) + 0.000367 * T +
             tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * log(T) + 0.014025 * T)};

  return Kokkos::exp(lnpsat) / dlc::P0;  // dimensionless psat
}

/* Calculate dimensionless Fkl and Fdl heat and vapour
diffusion factors in equation for radial growth of droplets
according to equations from "An Introduction To Clouds...."
(see note at top of file). fkl is first item of returned
pair, fdl is second. */
KOKKOS_FUNCTION
double diffusion_factor(const double press, const double temp, const double psat) {
  constexpr double A = 7.11756e-5;                             // coefficient for T^2 in T*[eq.7.24]
  constexpr double B = 4.38127686e-3;                          // coefficient for T in T*[eq.7.24]
  constexpr double LATENT_RGAS_V = DC::LATENT_V / DC::RGAS_V;  // for fkl diffusion factor calc
  constexpr double D = 4.012182971e-5;                         // constants in equation [eq.7.26]

  const auto TEMP = double{temp * dlc::TEMP0};
  const auto PRESS = double{press * dlc::P0};
  const auto PSAT = double{psat * dlc::P0};

  const auto THERMK =
      double{A * Kokkos::pow(TEMP, 2.0) + TEMP * B};  // K*TEMP with K from [eq.7.24] (for fkl)
  const auto DIFFUSE_V = double{(D / PRESS * Kokkos::pow(TEMP, 1.94)) /
                                DC::RGAS_V};  // 1/R_v * D_v from [eq 7.26] (for fdl)

  const auto fkl =
      double{(LATENT_RGAS_V / TEMP - 1.0) * DC::LATENT_V / (THERMK * dlc::F0)};  // fkl eqn [7.23]
  const auto fdl = double{TEMP / (DIFFUSE_V * PSAT) / dlc::F0};                  // fdl eqn [7.25]

  return dlc::Rho_l * (fkl + fdl);  // total constant from diffusion factors
}

#endif  // LIBS_SUPERDROPS_THERMODYNAMIC_EQUATIONS_HPP_
