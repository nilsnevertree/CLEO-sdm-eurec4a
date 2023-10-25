/*
 * ----- CLEO -----
 * File: microphysicalprocess.hpp
 * Project: superdrops
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Wednesday 25th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * Microphysical Process Concept as well as
 * helpers for creating structures that obey the
 * concept to model microphysics in SDM, 
 * eg. condensation or collision-coalescence
 * (see ConstTstepProcess struct)
 */

#ifndef MICROPHYSICALPROCESS_HPP
#define MICROPHYSICALPROCESS_HPP

#include <concepts>

#include <Kokkos_Core.hpp>

#include "../cleoconstants.hpp"
#include "../kokkosaliases.hpp"
#include "./superdrop.hpp"
#include "./state.hpp"
#include "./urbg.hpp"

template <typename P>
concept MicrophysicalProcess = requires(P p,
                                        const unsigned int t,
                                        const subviewd_supers supers,
                                        State &state,
                                        URBG<> &urbg)
/* concept for Microphysical Process is all types that
meet requirements (constraints) of these two timstepping
functions ()"on_step" and "next_step") as well as the
constraints on the "run_step" function */
{
  {
    p.next_step(t)
  } -> std::convertible_to<unsigned int>;
  {
    p.on_step(t)
  } -> std::same_as<bool>;
  {
    p.run_step(t, supers, state, urbg)
  } -> std::convertible_to<subviewd_supers>;
};

template <MicrophysicalProcess Microphys1,
          MicrophysicalProcess Microphys2>
struct CombinedMicrophysicalProcess
{
private:
  Microphys1 a;
  Microphys2 b;

public:
  CombinedMicrophysicalProcess(const Microphys1 a, const Microphys2 b)
      : a(a), b(b) {}

  KOKKOS_INLINE_FUNCTION
  unsigned int next_step(const unsigned int subt) const
   /* for combination of 2 microphysical processes,
   the next timestep is smaller out of the two possible */
  {
    const unsigned int t_a(a.next_step(subt));
    const unsigned int t_b(b.next_step(subt));

    return !(t_a < t_b) ? t_b : t_a; // return smaller of two unsigned ints (see std::min)
  }

  KOKKOS_INLINE_FUNCTION
  bool on_step(const unsigned int subt) const
  /* for combination of 2 microphysical processes,
  a tstep is on_step = true if either individual
  process is on_step = true */
  {
    return a.on_step(subt) || b.on_step(subt);
  }

  KOKKOS_INLINE_FUNCTION
  template <class DeviceType>
  void run_step(const unsigned int subt,
                const subviewd_supers supers,
                State &state,
                URBG<DeviceType> &urbg) const
  /* for combination of 2 proceses, each process
  is called sequentially */
  {
    supers = a.run_step(subt, supers, state, urbg);
    supers = b.run_step(subt, supers, state, urbg);
    return supers;
  }
};

auto operator>>(const MicrophysicalProcess auto a,
                const MicrophysicalProcess auto b)
/* define ">>" operator that combines
two Superdroplet Model Microphysical Processes */
{
  return CombinedMicrophysicalProcess(a, b);
}

struct NullMicrophysicalProcess
/* NullProcess does nothing at all
(is defined for a Monoid Structure) */
{
  KOKKOS_INLINE_FUNCTION
  unsigned int next_step(const unsigned int subt) const
  {
    return LIMITVALUES::uintmax;
  }
  
  KOKKOS_INLINE_FUNCTION
  bool on_step(const unsigned int subt) const
  {
    return false;
  }

  KOKKOS_INLINE_FUNCTION
  template <class DeviceType>
  void run_step(const unsigned int subt,
                const subviewd_supers supers,
                State &state,
                URBG<DeviceType> &urbg) const { return supers; }
};

template <typename F>
concept MicrophysicsFunc = requires(F f, const unsigned int subt,
                                    const subviewd_supers supers,
                                    State &state,
                                    URBG<> &urbg)
/* concept for all (function-like) types
(ie. types that can be called with some arguments)
that can be called by the run_step function in
ConstTstepMicrophysics (see below) */
{
  {
    f(subt, supers, state, urbg)
  } -> std::convertible_to<subviewd_supers>;
};

template <MicrophysicsFunc F>
struct ConstTstepMicrophysics
/* this structure is a type that satisfies the concept of
microphysical process in SDM and has a constant tstep
'interval'. It can be used to create a microphysical
processes with a constant timestep and microphysics 
determined by the MicrophysicsFunc type 'F' */
{
private:
  unsigned int interval;
  F do_microphysics;

public:
  ConstTstepMicrophysics(const unsigned int interval, const F f)
      : interval(interval), do_microphysics(f) {}

  KOKKOS_INLINE_FUNCTION
  unsigned int next_step(const unsigned int subt) const
  {
    return ((subt / interval) + 1) * interval;
  }

  KOKKOS_INLINE_FUNCTION
  bool on_step(const unsigned int subt) const
  {
    return subt % interval == 0;
  }

  KOKKOS_INLINE_FUNCTION
  template <class DeviceType>
  void run_step(const unsigned int subt,
                const subviewd_supers supers,
                State &state,
                URBG<DeviceType> &urbg) const
  {
    if (on_step(subt))
    {
      supers = do_microphysics(subt);
    }

    return supers;
  }
};

#endif // MICROPHYSICALPROCESS_HPP