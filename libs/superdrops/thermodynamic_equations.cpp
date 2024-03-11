/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: thermodynamic_equations.cpp
 * Project: superdrops
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Monday 11th March 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * implementation of functions that return
 * Left Hand Side of thermodynamic equations.
 * Equations referenced as (eqn [X.YY])
 * are from "An Introduction To Clouds From The
 * Microscale to Climate" by Lohmann, Luond
 * and Mahrt, 1st edition.
 */


#include "./thermodynamic_equations.hpp"

/**
 * @brief Calculate the equilibrium vapor pressure of water over liquid water,
 * i.e. the saturation pressure.
 *
 * Equation adapted from Bjorn Steven's "make_tetens" python function from his module
 * "moist_thermodynamics.saturation_vapour_pressures" available upon request on gitlab. Original
 * paper for formula is Murray, F. W. (1967) "On the Computation of Saturation Vapor Pressure",
 * Journal of Applied Meteorology and Climatology 6, 203–204.
 *
 * Note: function starts with conversion from dimentionless to real temperature [Kelvin],
 * TEMP = temp*Temp0, and returns dimensionless pressure from real psat = PSAT/P0.
 *
 * @param temp The (dimensionless) ambient temperature.
 * @return The (dimensionless) saturation pressure.
 */
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

/**
 * @brief Calculate the equilibrium vapor pressure of water over liquid water,
 * i.e. the saturation pressure.
 *
 * Equation adapted from Python module typhon.physics.thermodynamics.e_eq_water_mk with conversion
 * to real TEMP /K = temp*Temp0 and return dimensionless psat from real psat, psat = PSAT/P0.
 *
 * @param temp The (dimensionless) temperature.
 * @return The (dimensionless) saturation pressure.
 */
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

/**
 * @brief Calculate the sum of the heat and vapor diffusion factors for
 * condensation-diffusion growth equation.
 *
 * Calculate the sum of heat and vapor diffusion factors 'Fkl' and 'Fdl' respectively for
 * condensation-diffusion growth equation of droplet radius. Equations [X.YY] are from "An
 * Introduction To Clouds From The Microscale to Climate" by Lohmann, Luond and Mahrt, 1st edition.
 *
 * @param press The ambient pressure.
 * @param temp The ambient temperature.
 * @param psat The saturation pressure.
 * @return The (dimensionless) diffusion factor.
 */
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

  return dlc::Rho_l * (fkl + fdl);  // total constant from sum of diffusion factors
}
