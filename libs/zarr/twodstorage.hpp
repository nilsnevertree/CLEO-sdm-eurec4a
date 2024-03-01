/* Copyright (c) 2023 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: twodstorage.hpp
 * Project: zarr
 * Created Date: Sunday 22nd October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Monday 23rd October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 */

#ifndef LIBS_ZARR_TWODSTORAGE_HPP_
#define LIBS_ZARR_TWODSTORAGE_HPP_

#include <cassert>
#include <stdexcept>
#include <string>
#include <tuple>

#include "./fsstore.hpp"
#include "./singlevarstorage.hpp"
#include "./storehelpers.hpp"

/* 2D storage with dimensions [time, dim1] where
nobs is number of observation events (no. time outputs)
and ndim1 is the number of elements in 1st dimension
of 2-D data i.e. no. elements observed for each time.
For example, ndim1 could equal the number of gridboxes
an observer observes during 1 observation. Data for values
of time and dim1 could be output using a CoordinateStorage */
template <typename T>
struct TwoDStorage : SingleVarStorage<T> {
 private:
  const std::string dim1name;  // name of 1st dimension (e.g. "gbxindex")
  const size_t
      ndim1;        // number elements in 1st dimension (e.g. number of gridboxes that are observed)
  size_t ndim1obs;  // accumulated number of gridboxes that have been observed
  unsigned int nobs;  // accumulated number of output times that have been observed

  /* write data in buffer to a chunk in store alongside metadata jsons */
  void writechunk() {
    const std::string chunknum = std::to_string(this->chunkcount) + ".0";
    std::tie(this->chunkcount, this->bufferfill) = storehelpers::writebuffer2chunk(
        this->store, this->buffer, this->name, chunknum, this->chunkcount);

    writejsons();
  }

  /* write strictly required metadata to decode chunks (MUST).
  Assert also check 2D data dimensions is as expected */
  void writejsons() {
    assert((this->ndata == nobs * ndim1) && "1D data length matches 2D array size");
    assert((this->chunksize % ndim1 == 0.0) &&
           "chunks are integer multiple of 1st dimension of 2-D data");

    const auto n1str = std::to_string(ndim1);
    const auto nobstr = std::to_string(nobs);
    const auto nchstr = std::to_string(this->chunksize / ndim1);

    const auto shape("[" + nobstr + ", " + n1str + "]");
    const auto chunks("[" + nchstr + ", " + n1str + "]");
    const std::string dims = "[\"time\", \"" + dim1name + "\"]";
    this->zarrayjsons(shape, chunks, dims);
  }

  /* increment counts of number of observations of gridboxes, ngbxobs,
  and the number of observations of all gridboxes, nobs */
  void increment_ndim1obs() {
    ++ndim1obs;
    nobs = ndim1obs / ndim1;  // same as floor() for positive integers
  }

 public:
  TwoDStorage(FSStore &store, const unsigned int maxchunk, const std::string name,
              const std::string dtype, const std::string units, const double scale_factor,
              const std::string i_dim1name, const size_t i_ndim1)
      : SingleVarStorage<T>(store, storehelpers::good2Dchunk(maxchunk, i_ndim1), name, dtype, units,
                            scale_factor),
        dim1name(i_dim1name),
        ndim1(i_ndim1),
        ndim1obs(0),
        nobs(0) {}

  /* upon destruction write any data leftover in buffer
  to a chunk and write array's metadata to a .json file */
  ~TwoDStorage() {
    if (this->bufferfill != 0) {
      writechunk();
    }
  }

  void is_dim1(const size_t goodndim1, const std::string &goodname) const {
    if (ndim1 != goodndim1) {
      const std::string errmsg("ndim1 is" + std::to_string(ndim1) + ", but should be " +
                               std::to_string(goodndim1));
      throw std::invalid_argument(errmsg);
    }

    if (dim1name != goodname) {
      const std::string errmsg("name of dim1 is " + dim1name + ", but should be " + goodname);
      throw std::invalid_argument(errmsg);
    }
  }

  /* write val in the zarr store and then increment
  number of observations counts */
  void value_to_storage(const T val) {
    SingleVarStorage<T>::value_to_storage(val);
    increment_ndim1obs();
  }
};

/* 2D storage with dimensions [time, gbxindex] for
multiple variables in each gridbox over time. Variables
copied come in type V and how they are copied and
their metadata etc. is defined by the buffers type.
nobs is number of observation events (no. time outputs)
and ngbxs is the number of elements in 1st dimension of
2-D data i.e. no. gridboxes observed for each time */
template <typename Buffers, typename V>
struct TwoDMultiVarStorage {
 private:
  FSStore &store;  // file system store satisfying zarr store specificaiton v2

  const size_t
      chunksize;  // fixed size of array chunks (=max no. datapoints in buffer before writing)
  unsigned int chunkcount;   // number of chunks of array so far written to store
  unsigned int buffersfill;  // number of datapoints so far copied into buffer
  unsigned int ndata;        // number of data points that have been observed

  const char zarr_format = '2';  // storage spec. version 2
  const char order =
      'C';  // layout of bytes within each chunk of array in storage, can be 'C' or 'F'
  const std::string compressor = "null";  // compression of data when writing to store
  const std::string fill_value = "null";  // fill value for empty datapoints in array
  const std::string filters = "null";     // codec configurations for compression
  const std::string dtype;                // datatype stored in arrays

  Buffers buffers;  // buffers to hold state variables and then copy to store
  const size_t
      ngbxs;       // number elements in 1st dimension (e.g. number of gridboxes that are observed)
  size_t ngbxobs;  // accumulated number of gridboxes that have been observed
  unsigned int nobs;  // accumulated number of output times that have been observed

  /* write strictly required metadata to decode chunks (MUST).
  Assert also check 2D data dimensions is as expected */
  void writejsons() const {
    assert((ndata == nobs * ngbxs) && "1D data length matches 2D array size");
    assert((chunksize % ngbxs == 0.0) &&
           "chunks are integer multiple of 1st dimension of 2-D data");

    const auto n1str = std::to_string(ngbxs);
    const auto nobstr = std::to_string(nobs);
    const auto nchstr = std::to_string(chunksize / ngbxs);

    const auto shape("[" + nobstr + ", " + n1str + "]");
    const auto chunks("[" + nchstr + ", " + n1str + "]");

    const std::string metadata = storehelpers::metadata(zarr_format, order, shape, chunks, dtype,
                                                        compressor, fill_value, filters);

    buffers.writejsons(store, metadata);
  }

  /* write data from buffers into chunks in store,
  then reset buffersfill and write associated metadata */
  void writechunks() {
    std::tie(chunkcount, buffersfill) = buffers.writechunks(store, chunkcount);

    writejsons();
  }

  /* copy data to buffers */
  void copy2buffers(const V values) {
    std::tie(ndata, buffersfill) = buffers.copy2buffer(values, ndata, buffersfill);
  }

  /* increment counts of number of observations of gridboxes, ngbxobs,
  and the number of observations of all gridboxes, nobs */
  void increment_ngbxobs() {
    ++ngbxobs;
    nobs = ngbxobs / ngbxs;  // same as floor() for positive integers
  }

 public:
  TwoDMultiVarStorage(FSStore &store, const unsigned int maxchunk, const std::string dtype,
                      const size_t ngbxs, const std::string endname)
      : store(store),
        chunksize(storehelpers::good2Dchunk(maxchunk, ngbxs)),
        chunkcount(0),
        buffersfill(0),
        ndata(0),
        dtype(dtype),
        buffers(endname, chunksize),
        ngbxs(ngbxs),
        ngbxobs(0),
        nobs(0) {}

  /* upon destruction write any data leftover in buffer
  to a chunk and write array's metadata to a .json file */
  ~TwoDMultiVarStorage() {
    if (buffersfill != 0) {
      writechunks();
    }
  }

  /* write val in the zarr store. First copy it to a buffer,
  then write buffer to a chunk in the store when the number
  of values in the buffer reaches the chunksize */
  void values_to_storage(const V values) {
    if (buffersfill == chunksize) {
      writechunks();
    }

    copy2buffers(values);
    increment_ngbxobs();
  }
};

#endif  // LIBS_ZARR_TWODSTORAGE_HPP_
