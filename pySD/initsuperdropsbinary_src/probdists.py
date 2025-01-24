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


class MinXiDistrib:
    """probability of radius for a given probability distribution but with a
    minimum value such that xi>='xi_min'"""

    def __init__(self, probdistrib, xi_min):
        self.probdistrib = probdistrib
        self.xi_min = xi_min

    def __call__(self, radii, totxi):
        """returns probability for radii from a certain distribution or the
        probability such that xi has minimum value 'xi_min'"""
        prob = self.probdistrib(radii, totxi)
        prob_min = self.xi_min / totxi
        return np.where(prob < prob_min, prob_min, prob)


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

    def __call__(self, radii):
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
        return self._probdistrib(radii)

    def _probdistrib(self, radii):
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
        return self._probdistrib(radii)

    def _probdistrib(self, radii):
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
        return self._probdistrib(radii)

    def _probdistrib(self, radii):
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


class LogNormal(ProbabilityDistribution):
    """
    Probability of radius given by a log-normal distribution.

    Attributes
    ----------
    geometric_mean : float
        Geometric mean of the distribution.
    geometric_std_dev : float
        Geometric standard deviation of the distribution.
    scale_factor : float
        Scale factor for the distribution.

    Methods
    -------
    __call__(radii: np.ndarray, density: bool = False) -> np.ndarray
        Returns probability of each radius for a log-normal distribution.
    log_normal_distribution(t: np.ndarray, geometric_mean: float, geometric_std_dev: float, scale_factor: float) -> np.ndarray
        Compute the log-normal distribution from geometric mean and geometric standard deviation.
    """

    def __init__(
        self,
        geometric_mean: float,
        geometric_std_dev: float,
        scale_factor: float,
    ):
        self.geometric_mean = geometric_mean
        self.geometric_std_dev = geometric_std_dev
        self.scale_factor = scale_factor

    def __call__(self, radii: np.ndarray, density: bool = False) -> np.ndarray:
        """
        Returns probability of each radius for a log-normal distribution.

        Parameters
        ----------
        radii : np.ndarray
            Array of radii [m].
        density : bool, optional
            If True, returns the normalized probability density (default is False).

        Returns
        -------
        np.ndarray
            Probability of each radius in radii.
        """

        probs = self.log_normal_distribution(
            t=radii,
            geometric_mean=self.geometric_mean,
            geometric_std_dev=self.geometric_std_dev,
            scale_factor=self.scale_factor,
        )

        if density is True:
            return probs / np.nansum(probs)
        else:
            return probs

    def log_normal_distribution(
        self,
        t: np.ndarray,
        geometric_mean: float,
        geometric_std_dev: float,
        scale_factor: float,
    ) -> np.ndarray:
        """
        Compute the log-normal distribution from geometric mean and geometric standard deviation.

        Parameters
        ----------
        t : np.ndarray
            Array of radii [m].
        geometric_mean : float
            Geometric mean of the distribution.
        geometric_std_dev : float
            Geometric standard deviation of the distribution.
        scale_factor : float
            Scale factor for the distribution.

        Returns
        -------
        np.ndarray
            Probability of each radius in radii.
        """

        sigtilda = np.log(geometric_std_dev)
        mutilda = np.log(geometric_mean)

        norm = scale_factor / (np.sqrt(2 * np.pi) * t * sigtilda)
        exponent = -((np.log(t) - mutilda) ** 2) / (2 * sigtilda**2)

        return norm * np.exp(exponent)


class DoubleLogNormal(ProbabilityDistribution):
    """
    Probability of radius given by the superposition of two log-normal distributions.

    Attributes
    ----------
    geometric_mean1 : float
        Geometric mean of the first log-normal distribution.
    geometric_mean2 : float
        Geometric mean of the second log-normal distribution.
    geometric_std_dev1 : float
        Geometric standard deviation of the first log-normal distribution.
    geometric_std_dev2 : float
        Geometric standard deviation of the second log-normal distribution.
    scale_factor1 : float
        Scale factor for the first log-normal distribution.
    scale_factor2 : float
        Scale factor for the second log-normal distribution.

    Methods
    -------
    log_normal1 : LogNormal
        Returns the first log-normal distribution.
    log_normal2 : LogNormal
        Returns the second log-normal distribution.
    __call__(radii: np.ndarray, density: bool = False) -> np.ndarray
        Returns probability of each radius in radii derived from superposition of two log-normal distributions.
    """

    def __init__(
        self,
        geometric_mean1: float,
        geometric_mean2: float,
        geometric_std_dev1: float,
        geometric_std_dev2: float,
        scale_factor1: float,
        scale_factor2: float,
    ):
        self.geometric_mean1 = geometric_mean1
        self.geometric_mean2 = geometric_mean2
        self.geometric_std_dev1 = geometric_std_dev1
        self.geometric_std_dev2 = geometric_std_dev2
        self.scale_factor1 = scale_factor1
        self.scale_factor2 = scale_factor2

    @property
    def log_normal1(self) -> LogNormal:
        """
        Returns the first log-normal distribution.

        Returns
        -------
        LogNormal
            The first log-normal distribution.
        """
        return LogNormal(
            geometric_mean=self.geometric_mean1,
            geometric_std_dev=self.geometric_std_dev1,
            scale_factor=self.scale_factor1,
        )

    @property
    def log_normal2(self) -> LogNormal:
        """
        Returns the second log-normal distribution.

        Returns
        -------
        LogNormal
            The second log-normal distribution.
        """
        return LogNormal(
            geometric_mean=self.geometric_mean2,
            geometric_std_dev=self.geometric_std_dev2,
            scale_factor=self.scale_factor2,
        )

    def __call__(self, radii: np.ndarray, density: bool = False) -> np.ndarray:
        """
        Returns probability of each radius in radii derived from superposition of two log-normal distributions.

        Parameters
        ----------
        radii : np.ndarray
            Array of radii [m].
        density : bool, optional
            If True, returns the normalized probability density (default is False).

        Returns
        -------
        np.ndarray
            Probability of each radius in radii.
        """

        # Density needs to be False for the relation of the two distributions to be correct
        prob1 = self.log_normal1(radii=radii, density=False)
        prob2 = self.log_normal2(radii=radii, density=False)

        probs = prob1 + prob2

        if density is True:
            return probs / np.nansum(probs)
        else:
            return probs


class ClouddropsHansenGamma(ProbabilityDistribution):
    """probability of radius according to gamma distribution for
    shallow cumuli cloud droplets from Poertge et al. 2023"""

    def __init__(self, reff, nueff):
        self.reff = reff
        self.nueff = nueff

    def __call__(self, radii: np.ndarray) -> np.ndarray:
        return self._probdistrib(radii)

    def _probdistrib(self, radii):
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
        return self._probdistrib(radii)

    def _probdistrib(self, radii):
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
