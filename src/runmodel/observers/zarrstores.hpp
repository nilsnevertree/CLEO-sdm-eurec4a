// Author: Clara Bayley and Tobias Kölling
// File: zarr_stores.hpp
/* objects that can be used as stores obyeying the 
zarr storage specification version 2 (e.g. see FSStore)
https://zarr.readthedocs.io/en/stable/spec/v2.html */

#ifndef ZARRSTORES_HPP
#define ZARRSTORES_HPP

#include <string_view>
#include <span>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>

#include "../gridbox.hpp"
#include "superdrop_solver/superdrop.hpp"

template <typename Store>
struct StoreAccessor
/* functions for converting types (e.g. vectors of 
unsigned integers or doubles) into vectors of single bytes to 
write to store under a given key. Store can be anything that 
satisfies the zarr storage specifcaiton version 2 */
{
  Store &store;
  std::string_view key;

  StoreAccessor &operator=(std::span<const uint8_t> buffer)
    /* write range of memory representing uint8_ts to store */
  {
    store.write(key, buffer);
    return *this;
  }

  StoreAccessor &operator=(std::string_view buffer)
  /* reinterpret range of memory representing string as 
  a range of memory representing uint8_ts, then write to store */
  {
    return operator=(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t *>(buffer.data()),
        buffer.size()));
  }

  template <typename T>
  StoreAccessor &operator=(std::span<const T> buffer)
  /* re-interpret range of memory representing vector of type T as 
  a range of memory representing uint8_ts, then write to store */
  {
    return operator=(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t *>(buffer.data()),
        buffer.size() * sizeof(T)));
  }
};

class FSStore
/* A file system (with root in 'basedir' directory) obeying Zarr 
version 2 requirements for a Store. Store contins a series 
of key, values where values may be data arrays or groups in the store. 
data for a given key is written to the store via the functions 
in StoreAccessor */
{
private:
  const std::filesystem::path basedir;

public:
  FSStore(std::filesystem::path basedir) : basedir(basedir)
  {
    // initialize a zarr group (i.e. dataset)
    const unsigned int zarr_format = 2;    // storage spec. version 2
    const std::string zgroupjson = "{\"zarr_format\": " + std::to_string(zarr_format) + "}";
    (*this)[".zgroup"] = zgroupjson;

    // global metadata (optional)
    (*this)[".zattrs"] = "{\"creator\": \"Clara Bayley\", "
                         "\"title\": \"store for output of coupled SDM\"}";
  }

  StoreAccessor<FSStore> operator[](std::string_view key)
  {
    return {*this, key};
  }

  bool write(std::string_view key, std::span<const uint8_t> buffer);
  /* write function called by StoreAccessor once data has been 
  converted into a vector of unsigned integer types */
};

namespace storagehelper
/* namespace for generic helper functions used to
write a double to a buffer, a buffer to a chunk of an
array in a store, and an array's metadata to a store */
{
  
  template <typename V>
  inline void val2buffer(const V val, std::vector<V> &buffer, const int j)
  /* copy a type T (e.g. a double) called 'val',
  to appropriate buffer at index j */
  {
    buffer[j] = val;
  }

  inline void writebuffer2chunk(FSStore &store, std::vector<double> &buffer,
                               const std::string name, const std::string chunk_num)
  /* write buffer vector into attr's store at chunk no. 'kk', then
  replace contents of buffer with nans */
  {
    store[name + "/" + chunk_num].operator=<double>(buffer);
    std::fill(buffer.begin(), buffer.end(), std::nan(""));
  }

  inline void writebuffer2chunk(FSStore &store, std::vector<unsigned int> &buffer,
                               const std::string name, const std::string chunk_num)
  /* write buffer vector into attr's store at chunk no. 'kk', then
  replace contents of buffer with largest possible unsigned int
  (via setting unsigned int to -1) */
  {
    store[name + "/" + chunk_num].operator=<unsigned int>(buffer);
    std::fill(buffer.begin(), buffer.end(), -1);
  }

  inline void writebuffer2chunk(FSStore &store, std::vector<size_t> &buffer,
                               const std::string name, const std::string chunk_num)
  /* write buffer vector into attr's store at chunk no. 'kk', then
  replace contents of buffer with largest possible size_t int
  (via setting size_t type (i.e. long unsigned int) to -1) */
  {
    store[name + "/" + chunk_num].operator=<size_t>(buffer);
    std::fill(buffer.begin(), buffer.end(), -1);
  }

  inline std::string metadata(const unsigned int zarr_format,
                              const char order,
                              const std::string &shape,
                              const std::string &chunks,
                              const std::string &dtype,
                              const std::string &compressor,
                              const std::string &fill_value,
                              const std::string &filters)
  /* make string of metadata for an array in a zarr store */
  {
    const std::string metadata = "{"
                                 "\"shape\": " +
                                 shape + ", "
                                         "\"chunks\": " +
                                 chunks + ", "
                                          "\"dtype\": \"" +
                                 dtype + "\", "
                                         "\"order\": \"" +
                                 order + "\", "
                                         "\"compressor\": " +
                                 compressor + ", "
                                              "\"fill_value\": " +
                                 fill_value + ", "
                                              "\"filters\": " +
                                 filters + ", "
                                           "\"zarr_format\": " +
                                 std::to_string(zarr_format) +
                                 "}";
    return metadata;
  }

  inline std::string arrayattrs(const std::string &dims,
                              const std::string units = " ",
                              const double scale_factor = 1)
  /* make string of zattrs attribute information for an array in a zarr store */
  {
    std::ostringstream sfstr;
    sfstr << std::scientific << scale_factor;

    const std::string arrayattrs = "{\"_ARRAY_DIMENSIONS\": " + dims + ", "
                                                                       "\"units\": " +
                                   "\""+ units + "\", "
                                           "\"scale_factor\": " +
                                   sfstr.str() + "}";
    return arrayattrs;
  }

  inline void write_zarrarrayjsons(FSStore &store, const std::string name,
                                const std::string &metadata,
                                const std::string &arrayattrs)
  /* write .zarray and .zattr json files into store for the
  metadata of an array of a variable called 'name' */
  {
    // strictly required metadata to decode chunks (MUST)
    store[name + "/.zarray"] = metadata;

    // define dimension names of this array, to make xarray and netCDF happy
    // (not a MUST, ie. not strictly required, by zarr)
    //e.g. "{\"_ARRAY_DIMENSIONS\": [\"x\"]}";
    store[name + "/.zattrs"] = arrayattrs; 
  }
};

#endif // ZARRSTORES_HPP