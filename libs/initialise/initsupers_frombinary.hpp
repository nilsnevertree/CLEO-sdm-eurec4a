/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: initsupers_frombinary.hpp
 * Project: initialise
 * Created Date: Tuesday 17th October 2023
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
 * struct for superdroplets' initial conditions for CLEO SDM (e.g. superdroplet attributes)
 * by reading binary file. InitSupersFromBinary instance can be used by InitConds
 * struct as SuperdropInitConds type.
 */

#ifndef LIBS_INITIALISE_INITSUPERS_FROMBINARY_HPP_
#define LIBS_INITIALISE_INITSUPERS_FROMBINARY_HPP_

#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

#include "./initconds.hpp"
#include "./optional_config_params.hpp"
#include "./readbinary.hpp"
#include "superdrops/superdrop_attrs.hpp"

/* struct containing functions which return data
for the initial conditions needed to create
superdroplets e.g. via the CreateSupers struct */
struct InitSupersFromBinary {
 private:
  std::filesystem::path initsupers_filename;  // filename for some of superdrops' initial conditons
  size_t totnsupers;        // total number of superdroplets (in kokkos view on device initially)
  unsigned int nspacedims;  // number of spatial dimensions to model (0-D, 1-D, 2-D of 3-D)

  /* sets initial data for solutes as
  a single SoluteProprties instance */
  void init_solutes_data(InitSupersData &initdata) const;

  /* sets initial data in initdata using data read
  from a binary file called initsupers_filename */
  void initdata_from_binary(InitSupersData &initdata) const;

  /* copy data for vectors from binary file to initdata struct */
  void read_initdata_binary(InitSupersData &initdata, std::ifstream &file,
                            const std::vector<VarMetadata> &meta) const;

  /* check all the vectors in the initdata struct all
  have sizes consistent with one another. Include
  coords data in check if nspacedims != 0 */
  void check_initdata_sizes(const InitSupersData &initdata) const;

 public:
  explicit InitSupersFromBinary(const OptionalConfigParams::InitSupersFromBinaryParams &config)
      : initsupers_filename(config.initsupers_filename),
        totnsupers(config.totnsupers),
        nspacedims(config.nspacedims) {}

  auto get_totnsupers() const { return totnsupers; }

  auto get_nspacedims() const { return nspacedims; }

  /* data size returned is number of variables as
  declared by the metadata for the first variable
  in the initsupers file */
  size_t fetch_data_size() const;

  /* return InitSupersData created by reading a binary
  file and creating a SoluteProperties struct.
  Then check that the input data has the correct sizes. */
  void fetch_data(InitSupersData &initdata) const {
    init_solutes_data(initdata);
    initdata_from_binary(initdata);
    check_initdata_sizes(initdata);
  }
};

#endif  // LIBS_INITIALISE_INITSUPERS_FROMBINARY_HPP_
