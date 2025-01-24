"""
----- CLEO -----
File: attrsgen.py
Project: initsuperdropsbinary_src
Created Date: Friday 13th October 2023
Author: Clara Bayley (CB)
Additional Contributors:
-----
Last Modified: Wednesday 10th January 2024
Modified By: CB
-----
License: BSD 3-Clause "New" or "Revised" License
https://opensource.org/licenses/BSD-3-Clause
-----
Copyright (c) 2023 MPI-M, Clara Bayley
-----
File Description:
attrsgen generates multiple superdroplet
attributes given individual generators
"""

import numpy as np
import numpy.typing as npt
import xarray as xr
from typing import Tuple
from ..gbxboundariesbinary_src import read_gbxboundaries as rgrid
from ..initsuperdropsbinary_src import rgens
from .probdists import ProbabilityDistribution
from .dryrgens import DryRadiiGenerator
from .rgens import RadiiGenerator


class AttrsGenerator:
    """class for functions to generate attributes of
    superdroplets given the generators for independent
    attributes e.g. for radius and coord3 in substantation
    of class"""

    def __init__(
        self, radiigen, dryradiigen, xiprobdist, coord3gen, coord1gen, coord2gen
    ):
        self.radiigen = radiigen  # generates radius (solute + water)
        self.dryradiigen = dryradiigen  # generates dry radius (-> solute mass)
        self.xiprobdist = xiprobdist  # droplet size distribution (-> multiplicity)

        self.coord3gen = coord3gen  # generates spatial coordinate
        self.coord1gen = coord1gen
        self.coord2gen = coord2gen

        self.ncoordsgen = sum(x is not None for x in [coord3gen, coord2gen, coord1gen])

    def mass_solutes(self, radii, RHO_SOL):
        """return the mass [Kg] of the solute in superdroplets given their
        dry radii [m] and solute density [Kg m^3]"""

        dryradii = self.dryradiigen(radii)  # [m]
        dryradii = np.where(radii < dryradii, radii, dryradii)
        msols = 4.0 / 3.0 * np.pi * (dryradii**3) * RHO_SOL

        return msols  # [Kg]

    def multiplicities(self, radii, NUMCONC, samplevol):
        """Calculate the multiplicity of the dry radius of each
        superdroplet given it's probability such that the total number
        concentration [m^-3] of real droplets in the volume, vol, [m^3]
        is about 'numconc'. Raise an error if any of the calculated
        mulitiplicities are zero"""

        totxi = NUMCONC * samplevol
        prob = self.xiprobdist(radii, totxi)  # normalised prob distrib
        xi = np.rint(prob * totxi)

        if any(xi == 0):
            num = len(xi[xi == 0])
            errmsg = (
                "ERROR, "
                + str(num)
                + " out of "
                + str(len(xi))
                + " SDs"
                + " created with multiplicity = 0. Consider increasing numconc"
                + " or changing range of radii sampled."
            )
            raise ValueError(errmsg)

        return np.array(xi, dtype=np.uint)

    def check_coordsgen_matches_modeldimension(self, nspacedims):
        if nspacedims != self.ncoordsgen:
            errmsg = (
                str(self.ncoordsgen)
                + " coord generators specified "
                + "but nspacedims = "
                + str(nspacedims)
            )
            raise ValueError(errmsg)

    def check_totalnumconc(self, multiplicities, NUMCONC, samplevol):
        """check number concentration of real droplets calculated from
        multiplicities is same as input value for number conc. Also check
        total number of real droplets is within 10% of the expected
        value given the input number conc and sample volume"""

        nreals = np.rint(NUMCONC * samplevol)
        calcnreals = np.rint(np.sum(multiplicities))
        calcnumconc = np.rint(calcnreals / samplevol)

        if np.rint(NUMCONC) != calcnumconc:
            errmsg = (
                "total real droplet concentration"
                + " {:0g} != numconc, {:0g}".format(calcnumconc, NUMCONC)
            )
            raise ValueError(errmsg)

        if abs(nreals - calcnreals) > 0.001 * nreals:
            errmsg = "total no. real droplets, {:0g},".format(
                calcnreals
            ) + " not consistent with sample volume {:.3g} m^3".format(samplevol)
            raise ValueError(errmsg)

    def print_totalconc(self, multiplicities, radii, mass_solutes, RHO_SOL, samplevol):
        """print statement of total num conc and mass conc"""

        def totmass(radius, msol, RHO_SOL):
            """total mass of droplets represented by a superdroplet
            droplet totmass = mass of water + solute"""

            RHO_L = 998.203  # density of liquid water [kg/m^3]
            massconst = 4.0 / 3.0 * np.pi * radius * radius * radius * RHO_L
            density_factor = 1.0 - RHO_L / RHO_SOL
            totmass = msol * density_factor + massconst

            return totmass * 1000  # [g]

        numconc = np.sum(multiplicities) / samplevol / 1e6  # [cm^-3]
        totmass = np.sum(totmass(radii, mass_solutes, RHO_SOL) * multiplicities)
        massconc = totmass / samplevol  # [g/m^3]

        msg = (
            "--- total droplet concentration = "
            + "{:1g}cm^-3 => {:.1g}g/m^3".format(numconc, massconc)
            + ", in {:.3g}m^3 volume --- ".format(samplevol)
        )

        print(msg)

    def generate_attributes(
        self, nsupers, RHO_SOL, NUMCONC, gridboxbounds, isprint=False
    ):
        """generate superdroplets (SDs) attributes that have dimensions
        by calling the appropraite generating functions"""

        gbxvol = rgrid.calc_domainvol(
            gridboxbounds[0:2], gridboxbounds[2:4], gridboxbounds[4:]
        )  # [m^3]
        radii = self.radiigen(nsupers)  # [m]

        mass_solutes = self.mass_solutes(radii, RHO_SOL)  # [Kg]

        multiplicities = self.multiplicities(radii, NUMCONC, gbxvol)

        if nsupers > 0:
            self.check_totalnumconc(multiplicities, NUMCONC, gbxvol)
            if isprint:
                self.print_totalconc(
                    multiplicities, radii, mass_solutes, RHO_SOL, gbxvol
                )

        return multiplicities, radii, mass_solutes  # units [], [m], [Kg], [m]

    def generate_coords(self, nsupers, nspacedims, gridboxbounds):
        """generate superdroplets (SDs) attributes that have dimensions
        by calling the appropraite generating functions"""

        self.check_coordsgen_matches_modeldimension(nspacedims)

        coord3, coord1, coord2 = np.array([]), np.array([]), np.array([])

        if self.coord3gen:
            coord3range = [
                gridboxbounds[0],
                gridboxbounds[1],
            ]  # [min,max] coord3 to sample within
            coord3 = self.coord3gen(nsupers, coord3range)

            if self.coord1gen:
                coord1range = [
                    gridboxbounds[2],
                    gridboxbounds[3],
                ]  # [min,max] coord1 to sample within
                coord1 = self.coord1gen(nsupers, coord1range)

                if self.coord2gen:
                    coord2range = [
                        gridboxbounds[4],
                        gridboxbounds[5],
                    ]  # [min,max] coord2 to sample within
                    coord2 = self.coord2gen(nsupers, coord2range)

        return coord3, coord1, coord2  # units [m], [m], [m]


class AttrsGeneratorBinWidth(AttrsGenerator):
    """ "
    A class to generate attributes of superdroplets given the generators for independent
    attributes such as radius and coordinates.

    Attributes
    ----------
    radiigen : RadiiGenerator
        Generates radius.
    dryradiigen : DryRadiiGenerator
        Generates dry radius.
    xiprobdist : ProbabilityDistribution
        Droplet size distribution (multiplicity).
        Which can handle the use of bin widths.
        It should return a probability distribution normalised by radii.
        So in units of m^-3 m^-1.
        Then the true probability distribution is renormalised by the bin widths.
        See also nilsnevertree/CLEO-sdm-eurec4a#58
    coord3gen : callable
        Generates the third spatial coordinate.
    coord1gen : callable
        Generates the first spatial coordinate.
    coord2gen : callable
        Generates the second spatial coordinate.

    Methods
    -------
    multiplicities(radii, samplevol, bin_width, correct_for_zeros=True, probdists_kwargs=dict())
        Calculate the multiplicity of the super droplets for given radii and bin widths.
    generate_attributes(nsupers, RHO_SOL, NUMCONC, gridboxbounds)
        Generate superdroplets (SDs) attributes by calling the appropriate generating functions.

    """

    def __init__(
        self,
        radiigen: RadiiGenerator,
        dryradiigen: DryRadiiGenerator,
        xiprobdist: ProbabilityDistribution,
        coord3gen,
        coord1gen,
        coord2gen,
    ):
        self.radiigen = radiigen  # generates radius (solute + water)
        self.dryradiigen = dryradiigen  # generates dry radius (-> solute mass)
        self.xiprobdist = xiprobdist  # droplet size distribution (-> multiplicity)

        self.coord3gen = coord3gen  # generates spatial coordinate
        self.coord1gen = coord1gen
        self.coord2gen = coord2gen

        self.ncoordsgen = sum(x is not None for x in [coord3gen, coord2gen, coord1gen])

    def multiplicities(
        self,
        radii: npt.NDArray[np.float64],
        samplevol: float,
        bin_width: npt.NDArray[np.float64],
        correct_for_zeros: bool = True,
        probdists_kwargs: dict = dict(),
    ) -> npt.NDArray[np.uint]:
        """
        Calculate the multiplicity of the super droplets for
        given radii of each superdroplet and the corresponding ``bin_width``
        it represents.

        It assumes, that the probability distribution of the radii is
        normalised by radii (units of m^-3 m^-1).
        The true probability distribution is then renormalised by the given bin widths
        and uses it to determine the multiplicity of each superdroplet for the sample volume.

        - PSD = Pr * bin_width
        - xi = PSD * samplevol

        The integral of the PSD over the range of radii sampled should be
        a correct representation of the total number concentration.


        Notes
        -----
        If `correct_for_zeros` is True, it ensures that no multiplicity
        is zero by reducing the largest multiplicities and adding
        1 to the radii where the multiplicies are 0.

        Parameters
        ----------
        radii : np.ndarray
            Array of radii of the superdroplets.
        samplevol : float
            Volume of the sample in cubic meters.
        bin_width : Union[None, np.ndarray], optional
            Width of the bins for the probability distribution. If None, defaults to 1.
        correct_for_zeros : bool, optional
            If True, corrects for zero multiplicities to ernsure all radii ranges consist of xi at least 1.
            The total number concentration from the sampled distirbution is conserved.

        Returns
        -------
            np.ndarray
        Array of multiplicities for each superdroplet.

        Raises
        ------
        TypeError
        If `bin_width` is not None or an instance of np.ndarray.
        ValueError
        If any of the calculated multiplicities are zero.


        """

        def check_zero_multiplicities(xi):
            if any(xi == 0):
                num = len(xi[xi == 0])
                errmsg = (
                    "ERROR, "
                    + str(num)
                    + " out of "
                    + str(len(xi))
                    + " SDs"
                    + " created with multiplicity = 0. Consider increasing numconc"
                    + " or changing range of radii sampled."
                )
                raise ValueError(errmsg)

        # calculate the normalized probability distirbution
        prob = self.xiprobdist(radii, **probdists_kwargs)

        # if the wrong instance is supplied, raise a TypeError
        if isinstance(
            bin_width,
            (
                np.ndarray,
                xr.DataArray,
                float,
            ),
        ):
            bin_width = bin_width
        else:
            raise TypeError(
                "bin_width must be np.ndarray but is {}".format(type(bin_width))
            )

        # calculate mulitplicity by renormalising the probability distribution
        # with the bin_width and multiplying with the sample volume
        xi = prob * bin_width * samplevol

        # get the total number concentration from the sampled distribution
        xi = np.rint(xi)
        desired_xi_sum = int(np.sum(xi))
        # if smaller than the radii size, the distirbution can
        # never represent the total number concentration
        if desired_xi_sum <= np.size(radii):
            check_zero_multiplicities(xi)

        # if the multiplicities are zero, adjust them to be at least 1
        # by reducing the largest values of xi to ensure the total number
        # of droplets is conserved
        if correct_for_zeros is True:
            xi_zero = xi < 1
            number_xi_zero = int(np.sum(xi_zero))
            # print(f'Adjust {number_xi_zero} multiplicities to be at least 1.')

            if np.sum(xi_zero) > 0:
                # reduce xi by 1 for the largest values of xi
                # to ensure that the total number of droplets is conserved
                sort_indices = np.argsort(xi)
                xi[sort_indices[-number_xi_zero:]] -= 1
                xi[xi_zero] += 1

            xi = np.rint(xi)
            adjusted_xi_sum = int(np.sum(xi))

            # validate the total number concentration is conserved
            np.testing.assert_equal(
                adjusted_xi_sum,
                desired_xi_sum,
                err_msg="The total number concentration is not conserved.",
            )

        # final check for zero multiplicities
        check_zero_multiplicities(xi)

        # return an unsigned integer array
        return np.array(xi, dtype=np.uint)

    def generate_attributes(
        self,
        nsupers: int,
        RHO_SOL: float,
        NUMCONC: int,  # not used at all
        gridboxbounds: Tuple[float, float, float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate superdroplets (SDs) attributes that have dimensions
        by calling the appropraite generating functions.

        Parameters
        """

        gbxvol = rgrid.calc_domainvol(
            gridboxbounds[0:2], gridboxbounds[2:4], gridboxbounds[4:]
        )  # [m^3]
        if isinstance(self.radiigen, rgens.SampleLog10RadiiWithBinWidth):
            # The radii generator is a SampleLog10RadiiWithBinWidth
            # which returns the radii AND the bin width
            radii_result = self.radiigen(nsupers)  # [m]
            radii = radii_result[0]
            bin_width = radii_result[1]
        elif isinstance(self.radiigen, rgens.RadiiGenerator):
            # If no bin width is provided, default to 1
            # Which is equal to neglecting the bin width
            radii = self.radiigen(nsupers)  # [m]
            bin_width = np.ones_like(radii)
        else:
            raise TypeError(
                "radiigen must be an instance of RadiiGenerator but is {}".format(
                    type(self.radiigen)
                )
            )
        mass_solutes = self.mass_solutes(radii, RHO_SOL)  # [Kg]

        multiplicities = self.multiplicities(
            radii=radii,
            samplevol=gbxvol,
            bin_width=bin_width,
        )

        multiplicities = multiplicities

        # if nsupers > 0:
        #     self.check_totalnumconc(multiplicities, NUMCONC, gbxvol)
        #     self.print_totalconc(multiplicities, radii, mass_solutes, RHO_SOL, gbxvol)

        return multiplicities, radii, mass_solutes  # units [], [m], [Kg], [m]
