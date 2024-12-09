"""
Copyright (c) 2024 MPI-M, Clara Bayley


----- CLEO -----
File: dryrgens.py
Project: initsuperdropsbinary_src
Created Date: Wednesday 27th December 2023
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
various ways of generating dryradii of
superdroplets for their initial conditions
"""

import numpy as np


class DryRadiiGenerator:
    """
    method to generate superdroplet dryradii that are uniform
    """

    def __init__(self, dryradii: float):
        self.dryradii = dryradii

    def __call__(
        self,
        radii: np.ndarray,
    ) -> np.ndarray:
        return np.zeros_like(radii) + self.dryradii


class ScaledRadiiGen(DryRadiiGenerator):
    """method to generate superdroplet dryradii that
    are the radii divided by a scale factor 'sf'"""

    def __init__(self, scale_factor):
        self.sf = scale_factor

    def __call__(self, radii):
        """Returns dryradii for nsupers [m]"""

        return radii / self.sf  # units [m]
