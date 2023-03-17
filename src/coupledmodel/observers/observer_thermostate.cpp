// Author: Clara Bayley
// File: observer_thermostate.cpp
/* structs/classes to create a ThermoStateObserver that writes
data from thermostate into orthogonal multidimensional array(s) */

#include "observer_thermostate.hpp"

void ThermoIntoStore::copy2buffers(const ThermoState &state, const int j)
/* copy press, temp, qvap and qcond data in the state to buffers at index j */
{
  storagehelper::val2buffer(state.press, pressbuffer, j);
  storagehelper::val2buffer(state.temp, tempbuffer, j);
  storagehelper::val2buffer(state.qvap, qvapbuffer, j);
  storagehelper::val2buffer(state.qcond, qcondbuffer, j);
}

void ThermoIntoStore::writechunks(FSStore &store, const int chunkcount)
/* write buffer vector into attr's store at chunkcount
and then replace contents of buffer with std::nans */
{
  const std::string chunknum = std::to_string(chunkcount)+".0";
  storagehelper::writebuffer2chunk(store, pressbuffer, "press", chunknum);
  storagehelper::writebuffer2chunk(store, tempbuffer, "temp", chunknum);
  storagehelper::writebuffer2chunk(store, qvapbuffer, "qvap", chunknum);
  storagehelper::writebuffer2chunk(store, qcondbuffer, "qcond", chunknum);
}

void ThermoIntoStore::zarrayjsons(FSStore &store,
                                  const std::string &metadata) const
/* write same .zarray metadata to a json file for each thermostate array
in store alongside distinct .zattrs json files */
{
  const std::string dims = "[\"time\", \"gbxindex\"]";
  const std::string pressattrs = storagehelper::arrayattrs(dims, "hPa", dlc::P0/100);
  const std::string tempattrs = storagehelper::arrayattrs(dims, "K", dlc::TEMP0);
  const std::string qvapattrs = storagehelper::arrayattrs(dims);
  const std::string qcondattrs = storagehelper::arrayattrs(dims);

  storagehelper::write_zarrarrayjsons(store, "press", metadata, pressattrs);
  storagehelper::write_zarrarrayjsons(store, "temp", metadata, tempattrs);
  storagehelper::write_zarrarrayjsons(store, "qvap", metadata, qvapattrs);
  storagehelper::write_zarrarrayjsons(store, "qcond", metadata, qcondattrs);
}

void ThermoStateStorage::thermodata_to_storage(const ThermoState &state)
/* write thermo variables from a thermostate in arrays in the zarr store. 
First copy data to buffers, then write buffers to chunks in the store 
when the number of datapoints they contain reaches the chunksize */
{
  if (bufferfill == chunksize)
  {
    // write data in buffer to a chunk in store
    buffers.writechunks(store, chunkcount);
    ++chunkcount;
    bufferfill = 0;
  }

  // copy data from thermostate to buffers
  buffers.copy2buffers(state, bufferfill);
  ++bufferfill;
  ++ndata;
}