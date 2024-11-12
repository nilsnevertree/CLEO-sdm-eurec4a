"""
----- CLEO -----
File: probdists.py
Project: initsuperdropsbinary_src
Created Date: Wednesday 22nd November 2023
Author: Clara Bayley (CB)
Additional Contributors:
-----
Last Modified: Wednesday 27th December 2023
Modified By: CB
-----
License: BSD 3-Clause "New" or "Revised" License
https://opensource.org/licenses/BSD-3-Clause
-----
Copyright (c) 2023 MPI-M, Clara Bayley
-----
File Description:
Class calls' return normalised probability
of radii for various probability distributions
assuming bins are evenly spaced in log10(r)
"""

import numpy as np
from scipy import special
from typing import Tuple, List, Union


# Distibution functions


def standard_normal(radii: np.ndarray) -> np.ndarray:
    """Returns probability of each radius in radii according to
    standard normal distribution

    Parameters:
    -----------
    radii : np.ndarray
        array of radii [m]

    Returns:
    --------
    probs : np.ndarray
        probability of each radius in radii
    """

    probs = 1 / np.sqrt(2 * np.pi) * np.exp(-(radii**2) / 2)
    return probs / np.sum(probs)  # normalise so sum(prob) = 1


class ProbabilityDistribution:
    """probability of radius from a probability distribution"""

    def __init__(self):
        pass

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        """returns distribution for radii given by the
        distribution in distrib attribute"""

        raise NotImplementedError("method must be implemented in subclass")


class CombinedRadiiProbDistribs(ProbabilityDistribution):
    """probability of radius from the sum of several
    probability distributions"""

    def __init__(
        self,
        probdistribs: Tuple[ProbabilityDistribution],
        scalefacs: Union[np.ndarray, Tuple[float], List[float]],
    ):
        self.probdistribs = probdistribs
        self.scalefacs = scalefacs

        if len(scalefacs) != len(probdistribs):
            errmsg = "relative height of each probability distribution must be given"
            raise ValueError(errmsg)

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        """returns distribution for radii given by the
        sum of the distributions in probdistribs list"""

        probs = np.zeros(radii.shape)
        for distrib, sf in zip(self.probdistribs, self.scalefacs):
            probs += sf * distrib(radii)

        return probs / np.sum(probs)  # normalise so sum(prob) = 1


class DiracDelta(ProbabilityDistribution):
    """probability of radius nonzero if it is
    closest value in sample of radii to r0"""

    def __init__(self, r0):
        self.r0 = r0

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        """Returns probability of radius in radii sample for
        discrete version of dirac delta function centred on
        value of r in radii closest to r0. For each radius in radii,
        probability of that radius = 0 if it's not the closest value
        in radii to r0. If it is the closest, the probability is maximal
        (ie. prob = 1 and is then re-normalised such that sum of the
        probalilities over the sample = 1)"""

        if radii.any():
            diff = np.abs(radii - self.r0)

            probs = np.where(diff == np.min(diff), 1, 0)

            return probs / np.sum(probs)  # normalise so sum(prob) = 1
        else:
            return np.array([])


class VolExponential(ProbabilityDistribution):
    """probability of radius given by exponential in
    volume distribution as defined by Shima et al. (2009)"""

    def __init__(self, radius0, rspan):
        self.radius0 = radius0  # peak of volume exponential distribution [m]
        self.rspan = rspan

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        """Returns probability of eaach radius in radii according to
        distribution where probability of volume is exponential and bins
        for radii are evently spaced in ln(r).
        typical parameter values:
        radius0 = 30.531e-6 # [m]
        numconc = 2**(23) # [m^-3]"""

        rwdth = np.log(
            self.rspan[-1] / self.rspan[0]
        )  # assume equally spaced bins in ln(r)

        dn_dV = np.exp(-((radii / self.radius0) ** 3))  # prob density P(volume)

        probs = rwdth * radii**3 * dn_dV  # prob density * ln(r) bin width

        return probs / np.sum(probs)  # normalise so sum(prob) = 1


class LnNormal(ProbabilityDistribution):
    """probability of radius given by lognormal distribution
    as defined by section 5.2.3 of "An Introduction to clouds from
    the Microscale to Climate" by Lohmann, Luond and Mahrt and radii
    sampled from evenly spaced bins in ln(r).
    typical parameter values:
    geomeans = [0.02e-6, 0.2e-6, 3.5e-6] # [m]
    geosigs = [1.55, 2.3, 2]
    scalefacs = [1e6, 0.3e6, 0.025e6]
    numconc = 1e9 # [m^-3]"""

    def __init__(self, geomeans, geosigs, scalefacs):
        nmodes = len(geomeans)
        if nmodes != len(geosigs) or nmodes != len(scalefacs):
            errmsg = "parameters for number of lognormal modes is not consistent"
            raise ValueError(errmsg)
        else:
            self.nmodes = nmodes
            self.geomeans = geomeans
            self.geosigs = geosigs
            self.scalefacs = scalefacs

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        """Returns probability of each radius in radii derived
        from superposition of Logarithmic (in e) Normal Distributions"""

        probs = np.zeros(radii.shape)
        for n in range(self.nmodes):
            probs += self.lnnormaldist(
                radii, self.scalefacs[n], self.geomeans[n], self.geosigs[n]
            )

        return probs / np.sum(probs)  # normalise so sum(prob) = 1

    def lnnormaldist(self, radii, scalefac, geomean, geosig):
        """calculate probability of radii given the paramters of a
        lognormal dsitribution accordin to equation 5.8 of "An
        Introduction to clouds from the Microscale to Climate"
        by Lohmann, Luond and Mahrt"""

        sigtilda = np.log(geosig)
        mutilda = np.log(geomean)

        norm = scalefac / (np.sqrt(2 * np.pi) * sigtilda)
        exponent = -((np.log(radii) - mutilda) ** 2) / (2 * sigtilda**2)

        dn_dlnr = norm * np.exp(exponent)  # eq.5.8 [lohmann intro 2 clouds]

        return dn_dlnr


class LnNormalMassSpace(ProbabilityDistribution):
    """
    probability of mass concentration by radius following a lognormal distribution.
    Very similar to the probaility distribution of number concentration given by
    section 5.2.3 of "An Introduction to clouds from the Microscale to Climate" by Lohmann, Luond and Mahrt.
    Radii sampled from evenly spaced bins in ln(r).

    Parameters:
    -----------
    geomeans : list
        list of geometric means for each mode [m]


    """

    def __init__(
        self,
        geomean: float,
        geosig: float,
        scalefac: float,
    ):
        self.geomean = geomean
        self.geosig = geosig
        self.scalefac = scalefac

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        """Returns probability of each radius in radii derived
        from superposition of Logarithmic (in e) Normal Distributions"""

        probs = np.zeros(radii.shape)
        probs = self.__distribution__(radii, self.scalefac, self.geomean, self.geosig)

        return probs / np.sum(probs)  # normalise so sum(prob) = 1

    def __distribution__(
        self, radii: np.ndarray, geomean: float, geosig: float, scalefac: float
    ) -> np.ndarray:
        """calculate probability of radii given the paramters of a
        lognormal dsitribution according
        to the mass concentration probability distribution.
        This is an own implementation which shall help to constrain the total mass concentration.
        It help to follow the observed mass concentration distribution closer.
        It is similar to  equation 5.8 of "An Introduction to clouds from the Microscale to Climate"
        by Lohmann, Luond and Mahrt,

        BUT following the mass concentration distribution.
        Thus, to gain the number concentration distribution, we need to transform the probability to mass concentration probability.
        So we need to multiply by 1/(4/3 * pi * rho * r^3).
        But because we normalise the probability we can just multiply by r^{-3}.

        When creating the multiplicity of the Super-Droplets, one needs to carefully adjust the scale factor:
        For this if the scalefactor of the MSD is S, the scale factor of the number concentration distribution is:
        1/(4/3 * pi * rho) * S, with rho being the density of the material.

        Parameters
        ----------
        radii : np.ndarray
            array of radii [m]
        scalefac : float
            scale factor for the distribution
        geomean : float
            geometric mean of the distribution [m]
        geosig : float
            geometric standard deviation of the distribution

        Returns
        -------
        result : np.ndarray
            probability of each radius in radii following the mass concentration probability distribution
            its units are [m^-3 m^-1]
        """

        # we can use the expression based on the Standard normal distribution
        # to calculate the probability of the mass concentration
        # we need to transform the probability to mass concentration probability
        # so we need to multiply by 1/(4/3 * pi * r^3)
        # but because we normalise the probability we can just multiply by r^{-3}

        sigtilda = np.log(geosig)
        mutilda = np.log(geomean)

        result = (
            # adjustment to have the number concentration distribution
            radii ** {-3}
            # distribution of the mass concentration
            * scalefac
            * 1
            / (sigtilda * radii)
            * standard_normal(radii=((np.log(radii) - mutilda) / sigtilda))
        )
        return result


class ClouddropsHansenGamma(ProbabilityDistribution):
    """probability of radius according to gamma distribution for
    shallow cumuli cloud droplets from Poertge et al. 2023"""

    def __init__(self, reff, nueff):
        self.reff = reff
        self.nueff = nueff

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        """return gamma distribution for cloud droplets
        given radius [m] using parameters from Poertge
        et al. 2023 for shallow cumuli (figure 12).
        typical values:
        reff = 7e-6 #[m]
        nueff = 0.08 # []"""

        xp = (1 - 2 * self.nueff) / self.nueff
        n0const = (self.reff * self.nueff) ** (-xp)
        n0const = n0const / special.gamma(xp)

        term1 = radii ** ((1 - 3 * self.nueff) / self.nueff)
        term2 = np.exp(-radii / (self.reff * self.nueff))

        probs = n0const * term1 * term2  # dn_dr [prob m^-1]

        return probs / np.sum(probs)  # normalise so sum(prob) = 1


class RaindropsGeoffroyGamma(ProbabilityDistribution):
    """probability of radius given gamma distribution for
    shallow cumuli rain droplets from Geoffroy et al. 2014"""

    def __init__(self, nrain, qrain, dvol):
        self.nrain = nrain  # raindrop concentration [ndrops/m^3]
        self.qrain = qrain  # rainwater content [g/m^3]
        self.dvol = dvol  # volume mean raindrop diameter [m]

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        """returns probability of each radius according to a
        gamma distribution for rain droplets using parameters
        from Geoffroy et al. 2014 for precipitating shallow
        cumuli RICO (see figure 3 and equations 2,3 and 5).
        typical parameter values:
        nrain = 3 / 0.001 # [ndrops/m^3]
        qrain = 0.9 # [g/m^3]
        dvol = 800e-6 #[m]"""

        nu = 18 / ((self.nrain * self.qrain) ** 0.25)  # []
        lamda = (nu * (nu + 1) * (nu + 2)) ** (1 / 3) / self.dvol  # [m^-1]
        const = self.nrain * lamda**nu / special.gamma(nu)

        diam = 2 * radii  # [m]
        probs = const * diam ** (nu - 1) * np.exp(-lamda * diam)  # dn_dr [prob m^-1]

        return probs / np.sum(probs)  # normalise so sum(prob) = 1
