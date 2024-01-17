/*
 * ----- CLEO -----
 * File: contigraggedstorage.hpp
 * Project: zarr
 * Created Date: Monday 23rd October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Tuesday 24th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * File for Contiguous Ragged Array Storage
 * used to store superdroplet attributes
 * (see: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_contiguous_ragged_array_representation)
 * in a FFStore obeying zarr storage specification verion 2:
 * https://zarr.readthedocs.io/en/stable/spec/v2.html */

#ifndef CONTIGRAGGEDSTORAGE_HPP
#define CONTIGRAGGEDSTORAGE_HPP

#include <concepts>
#include <vector>
#include <string>
#include <tuple>

#include "./fsstore.hpp"
#include "./storehelpers.hpp"
#include "./superdropsbuffers.hpp"

template <SuperdropsBuffers Buffers>
class ContigRaggedStorage
/* Class for outputting Superdrop's data into zarr storage in
arrays of contigous ragged representation with 'chunkcount' number
of chunks that have a fixed chunksize. Works by filling buffers in
buffers with superdrop data and then writing these buffers
into chunks in their corresponding array stores when number of
datapoints copied to the buffers reaches chunksize. */
{
private:
  FSStore &store;         // file system store satisfying zarr store specificaiton v2
  const size_t chunksize; // fixed size of array chunks (=max no. datapoints in buffer before writing)

  std::vector<size_t> rgdcount;                       // count variable for contiguous ragged representation of arrays
  unsigned int rgdcount_chunkcount;                   // number of chunks of rgdcount array so far written to store
  unsigned int rgdcount_bufferfill;                   // number of rgdcount values so far copied into its buffer
  unsigned int rgdcount_ndata;                        // number of rgdcount values observed so far
  const std::string rgdcount_name = "rgd_totnsupers"; // name of raggedcount zarray in store
  const std::string rgdcount_dtype = "<u8";           // datatype of raggedcount variable

  Buffers buffers;          // buffers and their handler functions for wrting SD data to store
  unsigned int chunkcount;  // number of chunks of array so far written to store
  unsigned int buffersfill; // number of datapoints so far copied into buffer
  unsigned int ndata;       // number of data points that have been observed (= size of array written to store)

  const char zarr_format = '2';          // storage spec. version 2
  const char order = 'C';                // layout of bytes within each chunk of array in storage, can be 'C' or 'F'
  const std::string compressor = "null"; // compression of data when writing to store
  const std::string fill_value = "null"; // fill value for empty datapoints in array
  const std::string filters = "null";    // codec configurations for compression

  void buffers_writejsons()
  {
    // write strictly required metadata to decode chunks (MUST)
    const std::string dims = "[\"sdId\"]";
    const SomeMetadata md(zarr_format, order, ndata, chunksize,
                          compressor, fill_value, filters, dims);
    buffers.writejsons(store, md);
  }

  void buffers_writechunk()
  /* write data in buffers to chunks of zarrays in store
  and (re)write associated metadata for zarrays */
  {
    std::tie(chunkcount, buffersfill) =
        buffers.writechunk(store, chunkcount);

    buffers_writejsons();
  }

  template <typename T>
  void copy2buffers(const T &value)
  /* copy data from superdrop to buffer(s) and
  increment required counting variables */
  {
    std::tie(ndata, buffersfill) =
        buffers.copy2buffer(value, ndata, buffersfill);
  }

  void rgdcount_writejsons()
  /* write zarray jsons for array of rgdcount variable in store */
  {
    const std::string
        count_arrayattrs("{\"_ARRAY_DIMENSIONS\": [\"time\"],"
                         "\"sample_dimension\": \"superdroplets\"}");

    const std::string
        count_metadata = storehelpers::
            metadata(zarr_format, order, rgdcount_ndata, chunksize,
                     rgdcount_dtype, compressor, fill_value, filters);

    storehelpers::writejsons(store, rgdcount_name,
                             count_metadata,
                             count_arrayattrs);
  }

  void rgdcount_writechunk()
  /* write rgdcount data in buffers to a chunk of its
  zarray in store and (re)write its associated metadata */
  {
    std::tie(rgdcount_chunkcount, rgdcount_bufferfill) =
        storehelpers::writebuffer2chunk(store, rgdcount,
                                        rgdcount_name,
                                        rgdcount_chunkcount);

    rgdcount_writejsons();
  }

  void copy2rgdcount(const size_t raggedn)
  /* write raggedn into rgdcount buffer */
  {
    std::tie(rgdcount_ndata, rgdcount_bufferfill) =
        storehelpers::val2buffer<size_t>(raggedn, rgdcount,
                                         rgdcount_ndata,
                                         rgdcount_bufferfill);
  }

public:
  ContigRaggedStorage(FSStore &store,
                      const size_t imaxchunk,
                      const Buffers ibuffers)
      : store(store), chunksize(imaxchunk), rgdcount(chunksize),
        rgdcount_chunkcount(0), rgdcount_bufferfill(0),
        rgdcount_ndata(0), buffers(ibuffers),
        chunkcount(0), buffersfill(0), ndata(0)
  {
    buffers.set_buffer(chunksize);
  }

  ~ContigRaggedStorage()
  {
    if (buffersfill != 0)
    {
      buffers_writechunk();
    }

    if (rgdcount_bufferfill != 0)
    {
      rgdcount_writechunk();
    }
  }

  template <typename T>
  void data_to_raggedstorage(const T &value)
  /* write 'value' in contiguous ragged representation of an array
  in the zarr store. First copy data to buffer(s), then write buffer(s)
  to chunks in the store when the number of datapoints they contain
  reaches the chunksize */
  {
    if (buffersfill == chunksize)
    {
      buffers_writechunk();
    }

    copy2buffers(value);
  }

  void raggedarray_count(const size_t raggedn)
  /* add element 'raggedn' to rgdcount. 'raggedn' should be
  number of datapoints written to buffer(s) during one event.
  rgdcount is then count variable for contiguous ragged
  representation of arrays written to store via buffer(s). */
  {
    if (rgdcount_bufferfill == chunksize)
    {
      rgdcount_writechunk();
    }

    copy2rgdcount(raggedn);
  }
};

#endif // CONTIGRAGGEDSTORAGE_HPP
