// Author: Clara Bayley
// File: impliciteuler.cpp
/* Class implementing the implicit euler
  method for radial growth/shrink of each superdroplet
  due to condensation and diffusion of water
  vapour according to equations from "An Introduction
  To Clouds From The Microscale to Climate" by
  Lohmann, Luond and Mahrt, 1st edition." and
  Shima et al. 2009 */

#include "impliciteuler.hpp"

double ImplicitEuler::solve_condensation(const double s_ratio,
                                         const double akoh,
                                         const double bkoh,
                                         const double fkl,
                                         const double fdl,
                                         const double rprev) const
/* forward timestep previous radius 'rprev' by delt using an implicit
euler method to integrate the condensation/evaporation ODg. Implict
timestepping equation defined in section 5.1.2 of Shima et al. 2009
and is root of polynomial g(z) = 0, where z = [R_i(t+delt)]^squared.
Newton Raphson iterations are used to converge towards the root of
g(z) within the tolerances of an ImpIter instance. Tolerances,
maxium number of iterations and sub-timestepping are adjusted based on
on the uniqueness criteria of the polynomial g(z). Refer to section
5.1.2 Shima et al. 2009 and section 3.3.3 of Matsushima et al. 2023
for more details. */
{
  const double ffactor(dlc::Rho_l * (fkl + fdl));
    const double max_uniquedelt(
        2.5 * ffactor / akoh * std::pow(5.0 * bkoh / akoh, 1.5));
  const double ract_ratio(rprev * rprev * akoh / 3.0 / bkoh);
   
  if (delt <= max_uniquedelt)
  /* criteria for unique solution are met */
  {
    const double subdelt = delt/100.0;
    const ImpIter impit{miniters, subdelt, maxrtol, maxatol,
                        s_ratio, akoh, bkoh, ffactor};
    double subr(rprev); 
    for (int i=0; i<100; ++i)
    {
      double init_ziter(initialguess(subr, s_ratio, akoh, bkoh));
      subr = impit.newtonraphson_niterations(subr, init_ziter, "B");
    }
    return subr;
  }

  else if (s_ratio <= 1.0 && ract_ratio < 1.0)
  /* criteria for unique solution are met */
  {
    const ImpIter impit{miniters, delt, maxrtol, maxatol,
                        s_ratio, akoh, bkoh, ffactor};

    double init_ziter(initialguess(rprev, s_ratio, akoh, bkoh));
    return impit.newtonraphson_niterations(rprev, init_ziter, "A");
  } 

  else
  {
    /* In general there may be > 0 spurious solutions. Convergence is
    slower so always allow >= 3 Newton Raphson Iterations */
    const unsigned int niters(std::max(miniters, (unsigned int)3));
    const ImpIter impit{niters, delt, maxrtol, maxatol,
                        s_ratio, akoh, bkoh, ffactor};

    double init_ziter(initialguess(rprev, s_ratio, akoh, bkoh));
    return impit.newtonraphson_niterations(rprev, init_ziter, "C");
  }

}

double ImplicitEuler::initialguess(const double rprev,
                                   const double s_ratio,
                                   const double akoh,
                                   const double bkoh) const
/* returns appropriate initial value (ie. a reasonable guess) for
'ziter' to use as first iteration of newton raphson method in
rootfinding algorithm for timestepping condensation/evaporation ODE.
Criteria is as in SCALE-SDM for making initial guess >> (activation radius)^2
if supersaturation > activation supersaturation for given droplet */
{
  const double rprevsqrd(std::pow(rprev, 2.0));
  const double s_act(1 + std::sqrt(4.0 * std::pow(akoh, 3.0) / 27 / bkoh)); // activation supersaturation

  if (s_ratio > s_act)
  {
    const double bigrsqrd = std::pow(1e-3 / dlc::R0, 2.0); // large initial guess for radius of drop that should be activated
    return std::max(bigrsqrd, rprev);
  }

  return rprevsqrd;
}

double ImplicitEuler::initialguess_shima(const double rprev,
                                         const double s_ratio,
                                         const double akoh,
                                         const double bkoh) const
/* returns appropriate initial value (ie. a reasonable guess)
as in Shima's SCALE-SDM with 2 criteria for modifying guess 
from rprev^2. Second criteria is that initial guess >= 
(equilibrium radius when s_ratio=1)^2, 'r1sqrd' */
{
  const double rsqrd = initialguess(rprev, s_ratio, akoh, bkoh);
  const double r1sqrd(bkoh / akoh);
  return std::max(rsqrd, r1sqrd);
}

double ImplicitEuler::ImpIter::
    newtonraphson_niterations(const double rprev,
                              double ziter,
                              const std::string scenario) const
/* Timestep condensation ODE by delt given initial guess for ziter,
(which is usually radius^squared from previous timestep). Uses newton
raphson iterative method to find new value of radius that converges
on the root of the polynomial g(ziter) within the tolerances of the
ImpIter instance. Uniquesol method assumes that solution to g(ziter)=0
is unique and therefore Newton Raphson root finding algorithm converges
quickly. This means method can be used with comparitively large tolerances
and timesteps, and the maximum number of iterations is small. After
'niters' iterations, convergence criteria is tested and futher
iterations undertaken if not yet converged. */
{
  // perform 'niters' iterations
  double numerator(0.0);
  for (unsigned int iter = 0; iter < niters; ++iter)
  {
    /* perform one attempted iteration  ziter^(m) -> ziter^(m+1)
    for iteration m+1 starting at m=1 */
    numerator = ode_gfunc(rprev, ziter);
    const double denominator = ode_gfuncderivative(ziter);
    ziter -= ziter * numerator / denominator; // increment ziter
  }

  // perform upto 'iterlimit' further iterations if convergence test fails
  if (!(isnotconverged(ode_gfunc(rprev, ziter), numerator)))
  {
    return std::sqrt(ziter);
  }
  else
  {
    const unsigned int iterlimit(50); // maximum number of further iterations
    return newtonraphson_iteruntilconverged(iterlimit, rprev, ziter, scenario);
  }
}

double ImplicitEuler::ImpIter::
    newtonraphson_iteruntilconverged(const unsigned int iterlimit,
                                     const double rprev,
                                     double ziter,
                                     const std::string scenario) const
/*  Timestep condensation ODE by delt given initial guess for ziter,
(which is usually radius^squared from previous timestep). Uses
newton raphson iterative method to find new value of radius that
converges on the root of the polynomial g(ziter) within the tolerances
of the ImpIter instance. After every iteration, convergence criteria is
tested and error is raised if method does not converge within
'iterlimit' iterations. Otherwise returns new value for the radius
(which is the radius at timestep 't+subdelt'. Refer to section 5.1.2 Shima
et al. 2009 and section 3.3.3 of Matsushima et al. 2023 for more details. */
{
  bool do_iter(true);
  unsigned int iter(1);

  // perform newton raphson iterations if convergence test fails
  // and throw error if not converged within 'iterlimit' iterations
  while (do_iter)
  {
    if (iter <= iterlimit)
    {
      /* perform one attempted iteration  ziter^(m) -> ziter^(m+1)
      for iteration m+1 starting at m=1 and then test for convergence */
      const auto iterret = iterate_rootfinding_algorithm(rprev, ziter);
      do_iter = iterret.first;
      ziter = iterret.second;
      iter += 1;
    }
    else
    {
      const std::string err = "Newton Raphson Method did not converge "
                              "within " +
                              std::to_string(iterlimit + niters) +
                              " iterations, case: " + scenario + "\n";
      throw std::invalid_argument(err);
      break;
    }
  }

  return std::sqrt(ziter);
}

std::pair<bool, double> ImplicitEuler::ImpIter::
    iterate_rootfinding_algorithm(const double rprev, double ziter) const
/* function performs one iteration of Newton Raphson rootfinding
  method and returns updated value of radius^2 alongside a boolean that
  is false if algorithm has converged */
{
  // increment ziter
  const double numerator = ode_gfunc(rprev, ziter);
  const double denominator = ode_gfuncderivative(ziter);
  ziter = ziter * (1 - numerator / denominator);

  // test for next iteration
  const double newnumerator = ode_gfunc(rprev, ziter);
  const bool do_iter = isnotconverged(newnumerator, numerator);

  return std::pair<bool, double>{do_iter, ziter};
}

double ImplicitEuler::ImpIter::
    ode_gfunc(const double rprev, const double rsqrd) const
/* returns g(z) / z * delt for g(z) function used in root finding
Newton Raphson Method for dr/dt condensation / evaporation ODE.
ODE is for radial growth/shrink of each superdroplet due to
condensation and diffusion of water vapour according to
equations from "An Introduction To Clouds...."
(see note at top of file). Note: z = ziter = radius^2 */
{
  const double radius = std::sqrt(rsqrd);

  const double alpha(s_ratio - 1 - akoh / radius + bkoh / std::pow(radius, 3.0));
  const double beta(2.0 * subdelt / (rsqrd * ffactor));
  const double gamma(std::pow(rprev / radius, 2.0));

  return 1 - gamma - alpha * beta;
}

double ImplicitEuler::ImpIter::
    ode_gfuncderivative(const double rsqrd) const
/* dg(z)/dz * delt, where dg(z)/dz is derivative of g(z) with
respect to z=rsqrd. g(z) is polynomial to find root of using
Newton Raphson Method. */
{
  const double radius = std::sqrt(rsqrd);

  const double alpha(akoh / radius - 3.0 * bkoh / std::pow(radius, 3.0));
  const double beta(subdelt / (rsqrd * ffactor));

  return 1 - alpha * beta;
}