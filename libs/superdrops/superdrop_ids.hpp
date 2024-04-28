/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: superdrop_ids.hpp
 * Project: superdrops
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors: Tobias Kölling (TK)
 * -----
 * Last Modified: Tuesday 9th April 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Header file for structs for assigning super-droplets with identifiers (IDs). E.g. and ID may
 * be a unique number starting from 0, or non-existant (occupying no memory)
 */

#ifndef LIBS_SUPERDROPS_SUPERDROP_IDS_HPP_
#define LIBS_SUPERDROPS_SUPERDROP_IDS_HPP_

#include <Kokkos_Core.hpp>
#include <ostream>

/**
 * @brief Struct containing value of SD identity (8 bytes integer).
 */
struct IntID {
  size_t value; /**< Value of the SD identity. */

  /**
   * @brief Class for generating unique SD identity.
   */
  class Gen {
   public:
    /**
     * @brief Generate the next SD identity.
     *
     * _Note:_ This generator is not thread-safe (_id++ is undefined in a multi-threaded
     * environment).
     *
     * @return SD identity.
     */
    IntID next() { return {_id++}; }

    /**
     * @brief Generate the next SD identity using given value 'id'.
     *
     * _Note:_ This generator assumes the ID was thread-safe generated (i.e., is unique).
     *
     * @param id The value to use for generating the next SD identity.
     * @return SD identity.
     */
    KOKKOS_INLINE_FUNCTION IntID next(const size_t id) { return {id}; }

   private:
    size_t _id = 0; /**< Internal counter for generating SD identities. */
  };
};

/**
 * @brief Struct for non-existent (no memory) SD identity.
 */
struct EmptyID {
  /**
   * @brief Class for generating empty SD identity.
   */
  class Gen {
   public:
    /**
     * @brief Generate an empty SD identity.
     *
     * @return Empty SD identity.
     */
    KOKKOS_INLINE_FUNCTION EmptyID next() { return {}; }

    /**
     * @brief Generate an empty SD identity.
     *
     * @param kk A parameter possibly used for generating SD identity.
     * @return Empty SD identity.
     */
    KOKKOS_INLINE_FUNCTION EmptyID next(const unsigned int kk) { return {}; }
  };
};

/**
 * @brief Output stream operator for printing super-droplet identity when it is IntID type.
 *
 * Statement is of value of super-droplet identity, IntID::value.
 *
 * @param os Output stream.
 * @param id SD identity to print.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &os, const IntID &id) {
  os << id.value;
  return os;
}

/**
 * @brief Output stream operator for printing null statement when super-droplet
 * identity is instance of EmptyID (non-existent) type.
 *
 *
 * Null statement reads "(Undefined) No ID".
 *
 * @param os Output stream.
 * @param id SD identity to print.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &os, const EmptyID &id) {
  os << "(Undefined) No ID";
  return os;
}

#endif  // LIBS_SUPERDROPS_SUPERDROP_IDS_HPP_
