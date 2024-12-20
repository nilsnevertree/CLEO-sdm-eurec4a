"""
Copyright (c) 2024 MPI-M, Clara Bayley


----- CLEO -----
File: rgens.py
Project: initsuperdropsbinary_src
Created Date: Friday 13th October 2023
Author: Clara Bayley (CB)
Additional Contributors:
-----
Last Modified: Tuesday 7th May 2024
Modified By: CB
-----
License: BSD 3-Clause "New" or "Revised" License
https://opensource.org/licenses/BSD-3-Clause
-----
File Description:
various ways of generatring radii of superdroplets for their initial conditions
"""

import numpy as np
from typing import Union, Tuple, List


class RadiiGenerator:
    """
    Base class which return
    """

    def __init__(self):
        pass

    def __call__(
        self,
        nsupers: int,
    ) -> np.ndarray:
        return np.ones(nsupers)


class MonoAttrGen(RadiiGenerator):
    """method to generate superdroplets with an
    attribute all equal to attr0"""

    def __init__(self, attr0):
        self.attr0 = attr0

    def __call__(self, nsupers):
        """Returns attribute for nsupers all
        with the value of attr0"""

        if type(nsupers) == np.ndarray:
            nsupers = np.shape(nsupers)[0]

        attrs = np.full(nsupers, self.attr0)

        return attrs


class SampleLog10RadiiGen:
    """method to generate superdroplet radii by randomly
    sampling from bins that are linearly spaced in log10(r)
    between rspan[0] and rspan[1]"""

    def __init__(self, rspan):
        self.rspan = rspan

    def __call__(self, nsupers):
        """Returns radii for nsupers sampled from rspan [m]"""

        return self.generate_radiisample(nsupers)  # units [m]

    def generate_radiisample(self, nbins):
        """Divide rspan [m] into evenly spaced bins in log10(r).
        If edges=True, return values of radii at edges of bins.
        Else sample each bin randomly to obtain the radius
        of 'nsupers' no. of superdroplets"""

        if nbins:
            log10redgs = np.linspace(
                np.log10(self.rspan[0]), np.log10(self.rspan[1]), nbins + 1
            )  # log10(r) bin edges

            radii = self.randomlysample_log10rbins(nbins, log10redgs)
            return radii  # [m]
        else:
            return np.array([])

    def randomlysample_log10rbins(self, nbins, log10redgs):
        """given the bin edges, randomly sample each bin of
        log10(radius /m) and return the resultant radii [m]"""

        log10r_binwidth = (log10redgs[-1] - log10redgs[0]) / nbins

        randlog10deltar = np.random.uniform(low=0.0, high=log10r_binwidth, size=nbins)
        randlog10r = log10redgs[:-1] + randlog10deltar

        radii = 10 ** (randlog10r)

        return radii  # [m]


class SampleLog10RadiiWithBinWidth(RadiiGenerator):
    """method to generate superdroplet radii by randomly
    sampling from bins that are linearly spaced in log10(r)
    between rspan[0] and rspan[1]"""

    def __init__(self, rspan: Union[List[float], Tuple[float, float]]):
        self.rspan = rspan

    def __call__(self, nsupers: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns radii for nsupers sampled from rspan [m]"""

        radii, bin_width = self.generate_radiisample(nsupers)  # units [m]
        return radii, bin_width

    def generate_radiisample(self, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Divide rspan [m] into evenly spaced bins in log10(r).
        If edges=True, return values of radii at edges of bins.
        Else sample each bin randomly to obtain the radius
        of 'nsupers' no. of superdroplets"""

        if nbins:
            log10redgs = np.linspace(
                np.log10(self.rspan[0]), np.log10(self.rspan[1]), nbins + 1
            )  # log10(r) bin edges

            radii = self.randomlysample_log10rbins(nbins, log10redgs)
            edges = 10**log10redgs
            bin_width = edges[1:] - edges[:-1]
            return radii, bin_width  # [m]

        else:
            return np.array([]), np.array([])

    def randomlysample_log10rbins(
        self, nbins: int, log10redgs: np.ndarray
    ) -> np.ndarray:
        """given the bin edges, randomly sample each bin of
        log10(radius /m) and return the resultant radii [m]"""

        log10r_binwidth = (log10redgs[-1] - log10redgs[0]) / nbins

        randlog10deltar = np.random.uniform(low=0.0, high=log10r_binwidth, size=nbins)
        randlog10r = log10redgs[:-1] + randlog10deltar
        # randlog10r = log10redgs[:-1] + log10r_binwidth / 2

        radii = 10 ** (randlog10r)

        return radii  # [m]
