/*
 * ----- CLEO -----
 * File: fromfile_cartesian_dynamics.cpp
 * Project: coupldyn_fromfile
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Monday 6th November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * functionality for dynamics solver in CLEO
 * where coupling is one-way and dynamics
 * are read from file
 */

#include "./fromfile_cartesian_dynamics.hpp"

std::vector<double>
thermodynamicvar_from_binary(std::string_view filename)
/* open file called 'filename' and return vector
of doubles for first variable in that file */
{
  /* open file and read in the metatdata
  for all the variables in that file */
  std::ifstream file(open_binary(filename));
  std::vector<VarMetadata> meta(metadata_from_binary(file));

  /* read in the data for the 1st variable in the file */
  std::vector<double>
      thermovar(vector_from_binary<double>(file, meta.at(0)));

  return thermovar;
}

std::array<size_t, 3>
kijfromindex(const std::array<size_t, 3> &ndims,
             const size_t index)
/* return (k,i,j) indicies from idx for a flattened 3D array
with ndims [nz, nx, ny]. kij is useful for then getting
position in of a variable in a flattened array defined on
the faces of the same grid. E.g for the w velocity defined 
on z faces of the grid which therefore has dims [nz+1, nx, ny] */
{
  const size_t j = index / (ndims[0] * ndims[1]);
  const size_t k = index % ndims[0];
  const size_t i = index / ndims[0] - ndims[1] * j;

  return std::array<size_t, 3>{k, i, j};
}

void CartesianDynamics::increment_position()
/* updates positions to gbx0 in vector (for
acessing value at next timestep). Assumes domain
is decomposed into cartesian C grid with dimensions
(ie. number of gridboxes in each dimension) ndims */
{
  pos += ndims[0] * ndims[1] * ndims[2];
  pos_zface += (ndims[0] + 1) * ndims[1] * ndims[2];
  pos_xface += ndims[0] * (ndims[1] + 1) * ndims[2];
  pos_yface += ndims[0] * ndims[1] * (ndims[2] + 1);
}

CartesianDynamics::
    CartesianDynamics(const Config &config,
                      const std::array<size_t, 3> i_ndims)
    : wvel_zfaces(0), uvel_xfaces(0), vvel_yfaces(0),
      ndims(i_ndims),
      pos(0),
      pos_zface(0),
      pos_xface(0),
      pos_yface(0),
      press(thermodynamicvar_from_binary(config.press_filename)),
      temp(thermodynamicvar_from_binary(config.temp_filename)),
      qvap(thermodynamicvar_from_binary(config.qvap_filename)),
      qcond(thermodynamicvar_from_binary(config.qcond_filename)),
      get_wvel(nullwinds()), get_uvel(nullwinds()), get_vvel(nullwinds())
{
  std::cout << "\nFinished reading thermodynamics from binaries for:\n"
               "  pressure,\n  temperature,\n"
               "  water vapour mass mixing ratio,\n"
               "  liquid water mass mixing ratio,\n";

  set_winds(config);

  // check_thermodyanmics_vectorsizes(config.nspacedims, ndims, nsteps);
}

void CartesianDynamics::set_winds(const Config &config)
/* depending on nspacedims, read in data
for 1-D, 2-D or 3-D wind velocity components */
{
  const unsigned int nspacedims(config.nspacedims);

  switch (nspacedims)
  {
  case 0:
    std::cout << "0-D model has no wind data\n";
    break;

  case 1:
  case 2:
  case 3: // 1-D, 2-D or 3-D model
    const std::string windstr(
        set_winds_from_binaries(nspacedims,
                                config.wvel_filename,
                                config.uvel_filename,
                                config.vvel_filename));
    std::cout << windstr;

  default:
    throw std::invalid_argument("nspacedims for wind data is invalid");
  }
}

std::string CartesianDynamics::
    set_winds_from_binaries(const unsigned int nspacedims,
                            std::string_view wvel_filename,
                            std::string_view uvel_filename,
                            std::string_view vvel_filename)
/* Read in data from binary files for wind
velocity components in 1D, 2D or 3D model
and check they have correct size */
{
  std::string infostart(std::to_string(nspacedims) +
                        "-D model, wind velocity");

  std::string infoend;
  switch (nspacedims)
  {
  case 3: // 3-D model
    vvel_yfaces = thermodynamicvar_from_binary(vvel_filename);
    get_vvel = get_vvel_from_binary();
    infoend = ", u";
  case 2: // 3-D or 2-D model
    uvel_xfaces = thermodynamicvar_from_binary(uvel_filename);
    get_uvel = get_uvel_from_binary();
    infoend = ", v" + infoend;
  case 1: // 3-D, 2-D or 1-D model
    wvel_zfaces = thermodynamicvar_from_binary(wvel_filename);
    get_wvel = get_wvel_from_binary();
    infoend = "w" + infoend;
  }

  return infostart + " = [" + infoend + "]\n";
}

CartesianDynamics::get_winds_func
CartesianDynamics::nullwinds()
/* nullwinds retuns an empty function 'func' that returns
{0.0, 0.0}. Useful for setting get_[X]vel[Y]faces functions
in case of non-existent wind component e.g. get_uvelyface
when setup is 2-D model (x and z only) */
{
  const auto func = [](const unsigned int ii)
  { return std::pair<double, double>{0.0, 0.0}; };

  return func;
}

CartesianDynamics::get_winds_func
CartesianDynamics::get_wvel_from_binary()
/* set function for retrieving wvel defined at zfaces of
a gridbox with index 'gbxindex' and return vector
containting wvel data from binary file */
{
  const auto func = [&](const unsigned int gbxindex)
  {
    const auto kij = kijfromindex(ndims, (size_t)gbxindex); // [k,i,j] of gridbox centre on 3D grid
    const size_t nzfaces(ndims[0] + 1);                     // no. z faces to same 3D grid

    size_t lpos(ndims[1] * nzfaces * kij[2] + nzfaces * kij[1] + kij[0]); // position of z lower face in 1D wvel vector
    lpos += pos_zface;
    const size_t uppos(lpos + 1); // position of z upper face

    return std::pair(wvel_zfaces.at(lpos), wvel_zfaces.at(uppos));
  };

  return func;
}