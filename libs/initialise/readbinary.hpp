/*
 * ----- CLEO -----
 * File: readbinary.hpp
 * Project: initialise
 * Created Date: Monday 30th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Tuesday 31st October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * tools for reding binary initialisation
 * file e.g. for making gridbox maps or
 * SD initial conditions */


#ifndef READBINARY_HPP
#define READBINARY_HPP

#include <string>
#include <string_view>
#include <iostream>
#include <ios>
#include <fstream>
#include <vector>
#include <stdexcept>

struct GblMetadata
/* Global Metadata is 4 unsigned integers at very
start of binary (used to declare layout of binary file)
as well as string describing data in file */
{
  unsigned int d0byte;        // position of first byte of data
  unsigned int charbytes;     // no. bytes of global metadata chars (in string)
  unsigned int nvars;         // no. variables in file
  unsigned int mbytes_pervar; //  no. bytes of metadata per variable
  std::string metastr;        // description of file contents

  GblMetadata(std::ifstream &file);

  std::string read_global_metastring(std::ifstream &file,
                                     const int off) const;
};

struct VarMetadata
/* metadata in file related to a
particular variable (vaR) in that file */
{
  unsigned int b0;     // first byte in file containing this var's data
  unsigned int bsize;  // size in bytes of 1 datapoint of this var
  unsigned int nvar;   // no. datapoints of this var
  char vtype;          // char indicating type of this var
  char units;          // char indicating units once data multiplied by scale_factor
  double scale_factor; // scale factor to re-dimensionalise data

  VarMetadata(){};

  VarMetadata(std::ifstream &file, const int off);
};

std::ifstream open_binary(std::string_view filename);
/* open binary file for reading or raise error */

std::vector<VarMetadata> metadata_from_binary(std::ifstream &file);
/* Given a binary file that follows the correct layout,
read and print the global metadata string at the start of the file,
then return a vector containing the metadata that is specific to
each of the variables in the file */

void check_vectorsizes(const std::vector<size_t> &sizes);
/* raise error if values in vector 'sizes' are not the same. Useful
for checking if vectors are the same size e.g. for vectors of
SD attributes created from reading initSDsfile and used to
make InitSdsData object */

template <typename T>
void binary_into_buffer(std::ifstream &file,
                        std::vector<T> &buffer)
{
  file.read(reinterpret_cast<char *>(buffer.data()),
            buffer.size() * sizeof(T));
}

template <typename T>
std::vector<T> vector_from_binary(std::ifstream &file,
                                  const VarMetadata &varmeta)
/* return vector of data read from ifstream file for
one variable in a binary file given that variable's
metadata is given by the VarMetadata instance, 'varmeta' */
{
  file.seekg(varmeta.b0, std::ios::beg);
  std::vector<T> vardata(varmeta.nvar, 0);
  binary_into_buffer<T>(file, vardata);

  return vardata; // data for variable in binary file given it's metadata
}


#endif // READBINARY_HPP
