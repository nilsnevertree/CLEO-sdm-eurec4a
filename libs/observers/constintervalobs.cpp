/*
 * ----- CLEO -----
 * File: constintervalobs.cpp
 * Project: observers
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Monday 16th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * Observer Concept and related structures for various ways
 * of observing (outputing data from) CLEO.
 * An example of an observer is printing some data
 * from a gridbox's thermostate to the terminal
 */


#include "./constintervalobs.hpp"

void ConstIntervalObserver::observe_gbxs(const unsigned int t_mdl,
                                    const viewh_constgbx h_gbxs) const
{
  std::cout << "obs @ t = " << t_mdl << "\n";
}