/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: yac_cartesian_dynamics.cpp
 * Project: coupldyn_yac
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Sunday 17th December 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * functionality for dynamics solver in CLEO
 * where coupling is one-way and dynamics
 * are read from file
 */

#include "coupldyn_yac/yac_cartesian_dynamics.hpp"
#include <iostream>

extern "C" {
  #include "yac_interface.h"
}

/* open file called 'filename' and return vector
of doubles for first variable in that file */
std::vector<double> thermodynamicvar_from_binary(std::string_view filename) {
  /* open file and read in the metatdata
  for all the variables in that file */
  std::ifstream file(open_binary(filename));
  std::vector<VarMetadata> meta(metadata_from_binary(file));

  /* read in the data for the 1st variable in the file */
  std::vector<double> thermovar(vector_from_binary<double>(file, meta.at(0)));

  return thermovar;
}

/* return (k,i,j) indicies from idx for a flattened 3D array
with ndims [nz, nx, ny]. kij is useful for then getting
position in of a variable in a flattened array defined on
the faces of the same grid. E.g for the w velocity defined
on z faces of the grid which therefore has dims [nz+1, nx, ny] */
std::array<size_t, 3> kijfromindex(const std::array<size_t, 3> &ndims, const size_t index) {
  const size_t j = index / (ndims[0] * ndims[1]);
  const size_t k = index % ndims[0];
  const size_t i = index / ndims[0] - ndims[1] * j;

  return std::array<size_t, 3>{k, i, j};
}

/* updates positions to gbx0 in vector (for
acessing value at next timestep). Assumes domain
is decomposed into cartesian C grid with dimensions
(ie. number of gridboxes in each dimension) ndims */
void CartesianDynamics::increment_position() {
  pos += ndims[0] * ndims[1] * ndims[2];
  pos_zface += (ndims[0] + 1) * ndims[1] * ndims[2];
  pos_xface += ndims[0] * (ndims[1] + 1) * ndims[2];
  pos_yface += ndims[0] * ndims[1] * (ndims[2] + 1);
}

void CartesianDynamics::receive_fields_from_yac() {
  int info, error;
  double * yac_raw_data = NULL;

  yac_raw_data = press.data();
  yac_cget(pressure_yac_id, 1, &yac_raw_data, &info, &error);

  yac_raw_data = temp.data();
  yac_cget(temp_yac_id, 1, &yac_raw_data, &info, &error);

  yac_raw_data = qvap.data();
  yac_cget(qvap_yac_id, 1, &yac_raw_data, &info, &error);

  yac_raw_data = qcond.data();
  yac_cget(qcond_yac_id, 1, &yac_raw_data, &info, &error);

  yac_raw_data = united_edge_data.data();
  yac_cget(hor_wind_velocities_yac_id, 1, &yac_raw_data, &info, &error);

  int lower_index, upper_index;
  lower_index = upper_index = 0;
  std::vector<double> * target;
  for (size_t lat_index = 0; lat_index < vertex_latitudes.size() * 2 - 1; lat_index++) {
        if (lat_index % 2 == 0) {
            target = &uvel;
            lower_index = upper_index;
            upper_index = lower_index + 30;
        } else {
            target = &wvel;
            lower_index = upper_index;
            upper_index = lower_index + 31;
        }
        target->insert(target->end(),
                       united_edge_data.begin() + lower_index,
                       united_edge_data.begin() + upper_index);
  }
}

CartesianDynamics::CartesianDynamics(const Config &config, const std::array<size_t, 3> i_ndims,
                                     const unsigned int nsteps)
    : ndims(i_ndims),
      pos(0),
      pos_zface(0),
      pos_xface(0),
      pos_yface(0),
      get_wvel(nullwinds()),
      get_uvel(nullwinds()),
      get_vvel(nullwinds()) {
  std::cout << "\n--- coupled cartesian dynamics from file ---\n";

  yac_cinit();

  yac_cdef_calendar(YAC_PROLEPTIC_GREGORIAN);
  yac_cdef_datetime("1850-01-01T00:00:00", "1850-12-31T00:00:00");

  // --- Component definition ---
  std::string component_name = "cleo";
  int component_id = -1;
  yac_cdef_comp(component_name.c_str(), &component_id);

  // --- Grid definition ---
  int grid_id = -1;
  std::string cleo_grid_name = "cleo_grid";

  int total_cells[2]      = {ndims[0], ndims[1]};
  int total_vertices[2]   = {ndims[0] + 1, ndims[1] + 1};
  int cyclic_dimension[2] = {0, 0};
  int total_edges         = ndims[0] + ndims[1] + 2 * ndims[0] * ndims[1];

  vertex_longitudes           = std::vector<double>(ndims[0] + 1, 0);
  vertex_latitudes            = std::vector<double>(ndims[1] + 1, 0);
  auto cell_center_longitudes = std::vector<double>(ndims[0]);
  auto cell_center_latitudes  = std::vector<double>(ndims[1]);
  std::vector<double> edge_centers_longitudes;
  std::vector<double> edge_centers_latitudes;

  // Defines the vertex longitude and latitude values in radians for grid creation
  // The values are later permuted by YAC to generate all vertex coordinates
  for (size_t i = 0; i < vertex_longitudes.size(); i++) {
    vertex_longitudes[i] = i * (2 * std::numbers::pi / (ndims[0] + 1));
    if (i < vertex_latitudes.size())
      vertex_latitudes[i] = (-0.5 * std::numbers::pi) +
                            (i + 1) * (std::numbers::pi / (ndims[1] + 2));
  }

  // Defines a regular 2D grid
  yac_cdef_grid_reg2d(cleo_grid_name.c_str(), total_vertices,
                      cyclic_dimension, vertex_longitudes.data(),
                      vertex_latitudes.data(), &grid_id);

  // --- Point definitions ---

  // Defines the cell center longitude and latitude values in radians
  // The values are later permuted by YAC to generate all cell center coordinates
  for (size_t i = 0; i < cell_center_longitudes.size(); i++) {
    cell_center_longitudes[i] = vertex_longitudes[i] + std::numbers::pi / (ndims[0] + 1);
    cell_center_latitudes[i]  = vertex_latitudes[i] + std::numbers::pi / (2 * (ndims[1] + 2));
  }

  // Defines the edge center longitude and latitude values in radians.
  // Since it is not possible to generate edge center coordinates with a single
  // permutation of longitude and latitude values, usage of the
  // yac_cdef_points_unstruct is required. The call then takes x and y arrays,
  // with the actual radian coordinates for each edge center point. Therefore,
  // these arrays will have a size equal to the number of edges.
  for (size_t lat_index = 0; lat_index < vertex_latitudes.size() * 2 - 1; lat_index++) {
    if (lat_index % 2 == 0) {
      edge_centers_longitudes.insert(edge_centers_longitudes.end(),
                                     cell_center_longitudes.begin(),
                                     cell_center_longitudes.end());
      edge_centers_latitudes.insert(edge_centers_latitudes.end(),
                                    cell_center_longitudes.size(),
                                    vertex_latitudes[lat_index / 2]);
    } else {
      edge_centers_longitudes.insert(edge_centers_longitudes.end(),
                                     vertex_longitudes.begin(),
                                     vertex_longitudes.end());
      edge_centers_latitudes.insert(edge_centers_latitudes.end(),
                                    vertex_longitudes.size(),
                                    cell_center_latitudes[(lat_index - 1) / 2]);
    }
  }

  int cell_point_id = -1;
  int edge_point_id = -1;
  yac_cdef_points_reg2d(grid_id, total_cells, YAC_LOCATION_CELL,
                        cell_center_longitudes.data(), cell_center_latitudes.data(),
                        &cell_point_id);
  yac_cdef_points_unstruct(grid_id, total_edges, YAC_LOCATION_EDGE,
                        edge_centers_longitudes.data(), edge_centers_latitudes.data(),
                        &edge_point_id);

  // --- Interpolation stack ---
  int interp_stack_id;
  yac_cget_interp_stack_config(&interp_stack_id);
  yac_cadd_interp_stack_config_nnn(interp_stack_id, YAC_NNN_AVG, 1, 1.0);

  // --- Field definitions ---
  int num_point_sets = 1;
  int collection_size = 1;

  yac_cdef_field("pressure", component_id, &cell_point_id,
                 num_point_sets, collection_size, "PT1M",
                 YAC_TIME_UNIT_ISO_FORMAT, &pressure_yac_id);

  yac_cdef_field("temperature", component_id, &cell_point_id,
                 num_point_sets, collection_size, "PT1M",
                 YAC_TIME_UNIT_ISO_FORMAT, &temp_yac_id);

  yac_cdef_field("qvap", component_id, &cell_point_id,
                 num_point_sets, collection_size, "PT1M",
                 YAC_TIME_UNIT_ISO_FORMAT, &qvap_yac_id);

  yac_cdef_field("qcond", component_id, &cell_point_id,
                 num_point_sets, collection_size, "PT1M",
                 YAC_TIME_UNIT_ISO_FORMAT, &qcond_yac_id);

  yac_cdef_field("hor_wind_velocities", component_id, &edge_point_id,
                 num_point_sets, collection_size, "PT1M",
                 YAC_TIME_UNIT_ISO_FORMAT, &hor_wind_velocities_yac_id);


  // --- Field coupling definitions ---
  yac_cdef_couple("yac_reader", "yac_reader_grid", "pressure",
                  "cleo", "cleo_grid", "pressure",
                  "PT1M", YAC_TIME_UNIT_ISO_FORMAT, YAC_REDUCTION_TIME_NONE,
                  interp_stack_id, 0, 0);

  yac_cdef_couple("yac_reader", "yac_reader_grid", "temperature",
                  "cleo", "cleo_grid", "temperature",
                  "PT1M", YAC_TIME_UNIT_ISO_FORMAT, YAC_REDUCTION_TIME_NONE,
                  interp_stack_id, 0, 0);

  yac_cdef_couple("yac_reader", "yac_reader_grid", "qvap",
                  "cleo", "cleo_grid", "qvap",
                  "PT1M", YAC_TIME_UNIT_ISO_FORMAT, YAC_REDUCTION_TIME_NONE,
                  interp_stack_id, 0, 0);

  yac_cdef_couple("yac_reader", "yac_reader_grid", "qcond",
                  "cleo", "cleo_grid", "qcond",
                  "PT1M", YAC_TIME_UNIT_ISO_FORMAT, YAC_REDUCTION_TIME_NONE,
                  interp_stack_id, 0, 0);

  yac_cdef_couple("yac_reader", "yac_reader_grid", "hor_wind_velocities",
                  "cleo", "cleo_grid", "hor_wind_velocities",
                  "PT1M", YAC_TIME_UNIT_ISO_FORMAT, YAC_REDUCTION_TIME_NONE,
                  interp_stack_id, 0, 0);

  // --- End of YAC definitions ---
  yac_cenddef();

  press            = std::vector<double>(total_cells[0] * total_cells[1], 0);
  temp             = std::vector<double>(total_cells[0] * total_cells[1], 0);
  qvap             = std::vector<double>(total_cells[0] * total_cells[1], 0);
  qcond            = std::vector<double>(total_cells[0] * total_cells[1], 0);
  united_edge_data = std::vector<double>(total_edges, 0);

  receive_fields_from_yac();

  std::cout << "Finished setting up YAC for receiving:\n"
               "  pressure,\n  temperature,\n"
               "  water vapour mass mixing ratio,\n"
               "  liquid water mass mixing ratio,\n";

  set_winds(config);

  std::cout << "--- cartesian dynamics from YAC: success ---\n";
}

CartesianDynamics::~CartesianDynamics() { yac_cfinalize(); }

/* depending on nspacedims, read in data
for 1-D, 2-D or 3-D wind velocity components */
void CartesianDynamics::set_winds(const Config &config) {
  const auto nspacedims = (unsigned int)config.nspacedims;

  switch (nspacedims) {
    case 0:
      std::cout << "0-D model has no wind data\n";
      break;

    case 1:
    case 2:
    case 3:  // 1-D, 2-D or 3-D model
    {
      const std::string windstr(set_winds_from_binaries(
          nspacedims, config.wvel_filename, config.uvel_filename, config.vvel_filename));
      std::cout << windstr;
    } break;

    default:
      throw std::invalid_argument("nspacedims for wind data is invalid");
  }
}

/* Read in data from binary files for wind
velocity components in 1D, 2D or 3D model
and check they have correct size */
std::string CartesianDynamics::set_winds_from_binaries(const unsigned int nspacedims,
                                                       std::string_view wvel_filename,
                                                       std::string_view uvel_filename,
                                                       std::string_view vvel_filename) {
  std::string infostart(std::to_string(nspacedims) + "-D model, wind velocity");

  std::string infoend;
  switch (nspacedims) {
    case 3:  // 3-D model
      vvel_yfaces = thermodynamicvar_from_binary(vvel_filename);
      get_vvel = get_vvel_from_binary();
      infoend = ", u";
    case 2:  // 3-D or 2-D model
      get_uvel = get_uvel_from_yac();
      infoend = ", v" + infoend;
    case 1:  // 3-D, 2-D or 1-D model
      get_wvel = get_wvel_from_yac();
      infoend = "w" + infoend;
  }

  return infostart + " = [" + infoend + "]\n";
}

/* nullwinds retuns an empty function 'func' that returns
{0.0, 0.0}. Useful for setting get_[X]vel[Y]faces functions
in case of non-existent wind component e.g. get_uvelyface
when setup is 2-D model (x and z only) */
CartesianDynamics::get_winds_func CartesianDynamics::nullwinds() const {
  const auto func = [](const unsigned int ii) { return std::pair<double, double>{0.0, 0.0}; };

  return func;
}

/* set function for retrieving wvel defined at zfaces of
a gridbox with index 'gbxindex' and return vector
containting wvel data from binary file */
CartesianDynamics::get_winds_func CartesianDynamics::get_wvel_from_yac() const {
  const auto func = [&](const unsigned int gbxindex) {
    const auto kij =
        kijfromindex(ndims, static_cast<size_t>(gbxindex));  // [k,i,j] of gridbox centre on 3D grid
    const size_t nzfaces(ndims[0] + 1);                      // no. z faces to same 3D grid

    size_t lpos(ndims[1] * nzfaces * kij[2] + nzfaces * kij[1] +
                kij[0]);  // position of z lower face in 1D wvel vector
    const size_t uppos(lpos + 1);  // position of z upper face

    return std::pair(wvel.at(lpos), wvel.at(uppos));
  };

  return func;
}

/* returns vector of yvel retrieved from binary
file called 'filename' where uvel is defined on
the x-faces (coord1) of gridboxes */
CartesianDynamics::get_winds_func CartesianDynamics::get_uvel_from_yac() const {
  const auto func = [&](const unsigned int gbxindex) {
    const auto kij =
        kijfromindex(ndims, static_cast<size_t>(gbxindex));  // [k,i,j] of gridbox centre on 3D grid
    const size_t nxfaces(ndims[1] + 1);                      // no. x faces to same 3D grid

    size_t lpos(nxfaces * ndims[0] * kij[2] + ndims[0] * kij[1] +
                kij[0]);  // position of x lower face in 1D uvel vector
    const size_t uppos(lpos + ndims[0]);  // position of x upper face

    return std::pair(uvel.at(lpos), uvel.at(uppos));
  };

  return func;
}

/* returns vector of vvel retrieved from binary
file called 'filename' where vvel is defined on
the y-faces (coord2) of gridboxes */
CartesianDynamics::get_winds_func CartesianDynamics::get_vvel_from_binary() const {
  const auto func = [&](const unsigned int gbxindex) {
    const size_t lpos(static_cast<size_t>(gbxindex) +
                      pos_yface);                    // position of y lower face in 1D vvel vector
    const size_t uppos(lpos + ndims[1] * ndims[0]);  // position of x upper face

    return std::pair(vvel_yfaces.at(lpos), vvel_yfaces.at(uppos));
  };

  return func;
}
