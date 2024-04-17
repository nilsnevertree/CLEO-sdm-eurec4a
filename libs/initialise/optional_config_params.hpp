/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: optional_config_params.hpp
 * Project: initialise
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Wednesday 17th April 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Header file for members of Config struct which determine CLEO's optional configuration
 * parameters read from a config file.
 */

#ifndef LIBS_INITIALISE_OPTIONAL_CONFIG_PARAMS_HPP_
#define LIBS_INITIALISE_OPTIONAL_CONFIG_PARAMS_HPP_

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <limits>
#include <string>

// TODO(CB): check types of config params e.g. int maxchunk -> size_t maxchunk
// TODO(CB): use std::filesystem::path not string

namespace NaNVals {
inline double dbl() { return std::numeric_limits<double>::signaling_NaN(); };
inline unsigned int uint() { return std::numeric_limits<unsigned int>::signaling_NaN(); };
}  // namespace NaNVals

/**
 * @brief Struct storing optional configuration parameters for CLEO
 *
 * Optional means parameters have default values and therefore need not be set upon
 * construction. Default values are not intended to be used and may caused model errors at runtime.
 *
 */
struct OptionalConfigParams {
  /* read configuration file given by config_filename to set members of required configuration */
  explicit OptionalConfigParams(const std::filesystem::path config_filename);

  /* Condensation Runtime Parameters */
  struct DoCondensationParams {
    bool do_alter_thermo = false;         /**< true = condensation alters the thermodynamic state */
    unsigned int iters = NaNVals::uint(); /**< suggested no. iterations of Newton Raphson Method */
    double SUBTSTEP = NaNVals::dbl();     /**< smallest subtimestep in cases of substepping [s] */
    double rtol = NaNVals::dbl();         /**< relative tolerance for implicit Euler integration */
    double atol = NaNVals::dbl();         /**< abolute tolerance for implicit Euler integration */
  } condensation;

  /* Coupled Dynamics Runtime Parameters for FromFileDynamics */
  struct FromFileDynamicsParams {
    using fspath = std::filesystem::path;
    unsigned int nspacedims = NaNVals::uint(); /**< no. of spatial dimensions to model */
    fspath press = fspath();                   /**< name of file for pressure data */
    fspath temp = fspath();                    /**< name of file for temperature data */
    fspath qvap = fspath();                    /**< name of file for vapour mixing ratio data */
    fspath qcond = fspath();                   /**< name of file for liquid mixing ratio data */
    fspath wvel = fspath();                    /**< name of file for vertical (z) velocity data */
    fspath uvel = fspath();                    /**< name of file for horizontal x velocity data */
    fspath vvel = fspath();                    /**< name of file for horizontal y velocity data */
  } fromfiledynamics;

  /* Coupled Dynamics Runtime Parameters for CvodeDynamics */
  struct CvodeDynamicsParams {
    double P_INIT = NaNVals::dbl();    /**< initial pressure [Pa] */
    double TEMP_INIT = NaNVals::dbl(); /**< initial temperature [T] */
    double relh_init = NaNVals::dbl(); /**< initial relative humidity (%) */

    double W_AVG = NaNVals::dbl(); /**< average amplitude of sinusoidal w [m/s] (dP/dt ~ w*dP/dz) */
    double T_HALF = NaNVals::dbl(); /**< timescale for w sinusoid, tau_half = T_HALF/pi [s] */
    double rtol = NaNVals::dbl(); /**< relative tolerance for integration of [P, T, qv, qc] ODEs */
    double atol = NaNVals::dbl(); /**< absolute tolerances for integration of [P, T, qv, qc] ODEs */
  } cvodedynamics;
};

#endif  // LIBS_INITIALISE_OPTIONAL_CONFIG_PARAMS_HPP_
