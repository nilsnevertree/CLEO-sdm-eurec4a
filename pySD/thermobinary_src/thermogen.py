"""
Copyright (c) 2024 MPI-M, Clara Bayley


----- CLEO -----
File: thermogen.py
Project: thermobinary_src
Created Date: Monday 16th October 2023
Author: Clara Bayley (CB)
Additional Contributors:
-----
Last Modified: Wednesday 26th March 2025
Modified By: CB
-----
License: BSD 3-Clause "New" or "Revised" License
https://opensource.org/licenses/BSD-3-Clause
-----
File Description:
Various ways of generating fields for Temp, Pressure, Qvap and Qcond to read into CLEO
"""

import numpy as np
from scipy import integrate
import xarray as xr
from .. import cxx2py
from .create_thermodynamics import thermoinputsdict
from .windsgen import DryAdiabat2DFlowField
from ..gbxboundariesbinary_src import read_gbxboundaries as rgrid
from typing import Union, Tuple


def get_Mrratio_from_constants_filename(constants_filename):
    consts = cxx2py.read_cxxconsts_into_floats(constants_filename)
    mconsts = cxx2py.derive_more_floats(consts)

    return mconsts["Mr_ratio"]


def saturation_press(TEMP):
    """Calculate the equilibrium vapor pressure of water over
    liquid water ie. the saturation pressure (psat) given the
    temperature [K]. Equation from Bjorn Steven's "make_tetens"
    function in module "moist_thermodynamics.saturation_vapour_pressures"
    available on gitlab. Original paper "Murray, F. W. On the
    Computation of Saturation Vapor Pressure. Journal of Applied
    Meteorology and Climatology 6, 203–204 (1967)."""

    Aconst = 17.4146
    Bconst = 33.639
    TREF = 273.16  # Triple point temperature [K] of water
    PREF = 611.655  # Triple point pressure [Pa] of water

    if np.any(TEMP <= 0.0):
        raise ValueError("psat ERROR: T must be larger than 0K." + " T = " + str(TEMP))

    return PREF * np.exp(Aconst * (TEMP - TREF) / (TEMP - Bconst))  # [Pa]


def relh2qvap(press, temp, relh, Mr_ratio):
    """convert relative humidity [%] (relh) into vapour mass
    mixing ratio (qvap) given ambient temperature and pressure
    and ratio of molecular masses: vapour/air"""

    vapourpress = saturation_press(temp) * relh / 100.0  # [Pa]

    qvap = Mr_ratio * vapourpress / (press - vapourpress)  # dimensionless [Kg/kg]

    return qvap


def sratio2qvap(sratio, press, temp, Mr_ratio):
    psat = saturation_press(temp)

    qvap = Mr_ratio * sratio
    qvap = qvap / (press / psat - 1)

    return qvap


def qparams_to_qvap(method, params, Mr_ratio, PRESS, TEMP):
    """returns qvaps given list of qvaps, supersaturation ratios
    or relative humidities"""

    if method == "qvap":
        qparams = params
        return qparams

    elif method == "sratio":
        qparams = []
        for sratio in params:
            qparams.append(sratio2qvap(sratio, PRESS, TEMP, Mr_ratio))
        return qparams

    elif method == "relh":
        qparams = []
        for relh in params:
            qparams.append(relh2qvap(PRESS, TEMP, relh, Mr_ratio))
        return qparams

    else:
        raise ValueError("valid method not given to generate qvap")


def __attrs_if_dataarray__(
    da: Union[np.ndarray, xr.DataArray], attrs: dict = {}, name: str = ""
) -> Union[np.ndarray, xr.DataArray]:
    """
    This function adds attributes to a xr.DataArray if it is one, otherwise it does nothing.

    Parameters
    ----------
    da : Union[np.ndarray, xr.DataArray]
        The data array to add attributes to.
    attrs : dict, optional
        The attributes to add. Default is {}.
    name : str, optional
        The name of the data array, if not provided in the ``attrs`` dict. Default is "".

    Returns
    -------
    Union[np.ndarray, xr.DataArray]
        The data array with attributes added.
        Or the unmodified numpy array if it is not a xr.DataArray.
    """

    if isinstance(da, xr.DataArray):
        name = attrs.pop("name", name)
        da.attrs.update(attrs)
        da.name = name
        return da
    elif isinstance(da, np.ndarray):
        return da
    else:
        raise ValueError("da must be either a np.ndarray or xr.DataArray")


def saturation_vapour_pressure(
    temperature: Union[xr.DataArray, np.ndarray]
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the saturation vapour pressure over water for a given temperature.

    Parameters
    ----------
    temperature : xr.DataArray
        The temperature in Kelvin.

    Returns
    -------
    np.ndarray
        The saturation vapour pressure in Pa.
    """
    T = temperature
    A_w = 2.543e11  # Pa
    B_w = 5420  # K
    es = A_w * np.exp(-B_w / T)  # Pa (K/K) = Pa #type: ignore
    attrs = dict(
        name="saturation_vapour_pressure",
        units="Pa",
        long_name="Saturation vapour pressure",
        description="Saturation vapour pressure over water calculated from temperature.",
    )
    es = __attrs_if_dataarray__(da=es, attrs=attrs)
    return es


def specific_humidity_from_water_vapour_pressure(
    water_vapour_pressure: Union[xr.DataArray, np.ndarray],
    pressure: Union[xr.DataArray, np.ndarray],
    simplified: bool = False,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the specific humidity from the water vapour pressure and the pressure.
    This follows (2.80) from Introduction to Clouds: From the Microscale to Climate.

    Simplified version uses:
    q_v = (epsilon * e) /  p
    e = q_v * p / epsilon

    Non simplified version uses:
    q_v = (epsilon * e) / (p - e + epsilon * e)
    e = q_v * p / (epsilon + q_v - epsilon * q_v)


    Citation
    --------
    (2.80) from Introduction to Clouds: From the Microscale to Climate
    Ulrike Lohmann, Felix Lüönd, Fabian Mahrt, and Gregor Feingold
    ISBN: 978-1-107-01822-8 978-1-139-08751-3


    Parameters
    ----------
    water_vapour_pressure : xr.DataArray
        The water vapour pressure in Pa.
    pressure : xr.DataArray
        The pressure in Pa.

    Returns
    -------
    np.ndarray
        The specific humidity in kg/kg.
    """
    e = water_vapour_pressure
    p = pressure
    epsilon = 0.622
    if simplified:
        q_v = epsilon * e / p
    else:
        q_v = epsilon * e / (p - e + epsilon * e)
    return q_v


def specific_humidity_from_relative_humidity_temperature_pressure(
    relative_humidity: Union[np.ndarray, xr.DataArray],
    temperature: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    simplified: bool = False,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the specific humidity from the relative humidity, temperature and pressure.

    Parameters
    ----------
    relative_humidity : xr.DataArray
        The relative humidity in %.
    temperature : xr.DataArray
        The temperature in Kelvin.
    pressure : xr.DataArray
        The pressure in Pa.
    simplified : bool, optional
        If set to True, the simplified version is used.
        Default is False.

    Returns
    -------
    np.ndarray
        The specific humidity in kg/kg.
    """
    es = saturation_vapour_pressure(temperature)
    # calculate the vapour pressure
    e = relative_humidity * es / 100
    q_v = specific_humidity_from_water_vapour_pressure(e, pressure, simplified)

    attrs = dict(
        name="specific_humidity",
        units="kg/kg",
        long_name="Specific humidity",
        description="Specific humidity calculated from relative humidity, temperature and pressure.",
    )
    q_v = __attrs_if_dataarray__(da=q_v, attrs=attrs)
    return q_v


def potential_temperature_from_temperature_pressure(
    air_temperature: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    pressure_reference: Union[
        float, np.ndarray, xr.DataArray
    ] = 100000,  # default value used for drop sondes dataset
    R_over_cp: float = 0.286,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the potential temperature from the air temperature and the pressure.

    Parameters
    ----------
    air_temperature : xr.DataArray or xr.DataArray
        The air temperature in Kelvin.
    pressure : xr.DataArray  or xr.DataArray
        The pressure in Pa.
    pressure_reference : float
        The reference pressure in Pa.
    R_over_cp : float, optional
        The ratio of the gas constant of air to the specific heat capacity at
        constant pressure. Default is 0.286.

    Returns
    -------
    np.ndarray
        The potential temperature in Kelvin.
    """
    theta = air_temperature * (pressure_reference / pressure) ** R_over_cp
    attrs = dict(
        name="potential_temperature",
        units="K",
        long_name="Potential temperature",
        description="Potential temperature calculated from air temperature and pressure.",
    )
    theta = __attrs_if_dataarray__(da=theta, attrs=attrs)
    return theta


def temperature_from_potential_temperature_pressure(
    potential_temperature: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    pressure_reference: Union[
        float, np.ndarray, xr.DataArray
    ] = 100000,  # default value used for drop sondes dataset
    R_over_cp: float = 0.286,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the potential temperature from the air temperature and the pressure.

    Parameters
    ----------
    air_temperature : xr.DataArray or xr.DataArray
        The air temperature in Kelvin.
    pressure : xr.DataArray  or xr.DataArray
        The pressure in Pa.
    pressure_reference : float
        The reference pressure in Pa.
    R_over_cp : float, optional
        The ratio of the gas constant of air to the specific heat capacity at
        constant pressure. Default is 0.286.

    Returns
    -------
    np.ndarray
        The potential temperature in Kelvin.
    """
    factor = (pressure_reference / pressure) ** R_over_cp
    air_temperature = potential_temperature / factor
    attrs = dict(
        name="temperature",
        units="K",
        long_name="Ambient temperature",
        description="Ambient temperature calculated from potential temperature and pressure.",
    )
    air_temperature = __attrs_if_dataarray__(da=air_temperature, attrs=attrs)
    return air_temperature


def constant_winds(ndims, ntime, THERMODATA, WVEL, UVEL, VVEL):
    """add arrays to thermodata dictionary for winds, array are
    empty by default, or given constant values WVEl, UVEL and VVEL.
    Here, shape_[X]face = no. data for wind velocity component
    defined on gridbox [X] faces"""

    for VEL in ["WVEL", "UVEL", "VVEL"]:
        THERMODATA[VEL] = np.array([])

    if WVEL is not None:
        shape_zface = int((ndims[0] + 1) * ndims[1] * ndims[2] * ntime)
        THERMODATA["WVEL"] = np.full(shape_zface, WVEL)

        if UVEL is not None:
            shape_xface = int((ndims[1] + 1) * ndims[2] * ndims[0] * ntime)
            THERMODATA["UVEL"] = np.full(shape_xface, UVEL)

            if VVEL is not None:
                shape_yface = int((ndims[2] + 1) * ndims[0] * ndims[1] * ntime)
                THERMODATA["VVEL"] = np.full(shape_yface, VVEL)

    return THERMODATA


def divfree_flowfield2D(
    wmax, zlength, xlength, rhotilda_zfaces, rhotilda_xfaces, gbxbounds, ndims
):
    zfaces, xcens_z = rgrid.coords_forgridboxfaces(gbxbounds, ndims, "z")[0:2]
    zcens_x, xfaces = rgrid.coords_forgridboxfaces(gbxbounds, ndims, "x")[0:2]

    ztilda = zlength / np.pi
    xtilda = xlength / (2 * np.pi)
    wamp = 2 * wmax

    WVEL = wamp / rhotilda_zfaces
    WVEL = WVEL * np.sin(zfaces / ztilda) * np.sin(xcens_z / xtilda)

    UVEL = wamp / rhotilda_xfaces * xtilda / ztilda
    UVEL = UVEL * np.cos(zcens_x / ztilda) * np.cos(xfaces / xtilda)

    return WVEL, UVEL


class ConstUniformThermo:
    """create thermodynamics that's constant in time and uniform throughout the domain"""

    def __init__(
        self,
        PRESS,
        TEMP,
        qvap,
        qcond,
        relh=False,
        constants_filename="",
    ):
        self.PRESS = PRESS  # pressure [Pa]
        self.TEMP = TEMP  # temperature [T]

        if relh:
            Mr_ratio = get_Mrratio_from_constants_filename(constants_filename)
            self.qvap = relh2qvap(
                PRESS, TEMP, relh, Mr_ratio
            )  # water vapour content []
        else:
            self.qvap = qvap

        self.qcond = qcond  # liquid water content []

    def generate_thermo(self, gbxbounds, ndims, ntime):
        shape_cen = int(
            ntime * np.prod(ndims)
        )  # = no. data for var defined at gridbox centers

        THERMODATA = {
            "PRESS": np.full(shape_cen, self.PRESS),
            "TEMP": np.full(shape_cen, self.TEMP),
            "qvap": np.full(shape_cen, self.qvap),
            "qcond": np.full(shape_cen, self.qcond),
        }

        return THERMODATA


class Simple2TierRelativeHumidity:
    """create thermodynamics that's constant in time with (P,T,qc) uniform throughout the domain
    and with relative humidity uniform above and below Zbase"""

    def __init__(
        self,
        config_filename,
        constants_filename,
        PRESS,
        TEMP,
        qvapmethod,
        qvapparams,
        Zbase,
        qcond,
    ):
        inputs = thermoinputsdict(config_filename, constants_filename)

        self.PRESS = PRESS  # pressure [Pa]
        self.TEMP = TEMP  # temperature [T]
        self.qcond = qcond  # liquid water content []

        # determine qvap [below, above] z (cloud) base
        self.Zbase = Zbase
        qvaps = qparams_to_qvap(qvapmethod, qvapparams, inputs["Mr_ratio"], PRESS, TEMP)
        self.qvap_below, self.qvap_above = qvaps

        self.RGAS_DRY = inputs["RGAS_DRY"]
        self.RGAS_V = inputs["RGAS_V"]
        self.RHO0 = inputs["RHO0"]

    def generate_qvap_profile(self, zfulls):
        qvap = np.where(zfulls >= self.Zbase, self.qvap_above, self.qvap_below)

        return qvap

    def rhotilda(self, ZCOORDS):
        """returns dimensionless rho_dry profile for use in stream function"""
        PRESS, TEMP = self.hydrostatic_adiabatic_thermo(ZCOORDS)
        RHO_DRY = PRESS / ((self.RGAS_DRY + self.qvap * self.RGAS_V) * TEMP)
        rhotilda = RHO_DRY / self.RHO0

        return rhotilda

    def generate_thermo(self, gbxbounds, ndims, ntime):
        zfulls, xfulls, yfulls = rgrid.fullcoords_forallgridboxes(gbxbounds, ndims)

        qvap = self.generate_qvap_profile(zfulls)

        shape_cen = int(
            ntime * np.prod(ndims)
        )  # = no. data for var defined at gridbox centers
        THERMODATA = {
            "PRESS": np.full(shape_cen, self.PRESS),
            "TEMP": np.full(shape_cen, self.TEMP),
            "qvap": np.tile(qvap, ntime),
            "qcond": np.full(shape_cen, self.qcond),
        }

        return THERMODATA


class DryHydrostaticAdiabatic2TierRelH:
    """create thermodynamics that's constant in time and in hydrostatic equillibrium with
    a dry adiabat accounting for the mass of water vapour in the air.
    Equations derived from Arabas et al. 2015 (sect 2.1).
    Relative humidity like for Simple2TierRelativeHumidity exceptional moist layer possible
    to add within in a certain height range"""

    def __init__(
        self,
        config_filename,
        constants_filename,
        PRESSz0,
        THETA,
        qvapmethod,
        qvapparams,
        Zbase,
        qcond,
        moistlayer,
    ):
        inputs = thermoinputsdict(config_filename, constants_filename)

        ### parameters of profile ###
        self.PRESSz0 = PRESSz0  # pressure at z=0m [Pa]
        self.THETA = THETA  # (constant) dry potential temperature [K]
        self.qcond = qcond  # liquid mass mixing ratio []

        # determine qvap [below, above] z (cloud) base
        self.Zbase = Zbase
        self.qvapmethod, self.qvapparams = qvapmethod, qvapparams
        self.qvapz0 = qparams_to_qvap(
            qvapmethod, qvapparams, inputs["Mr_ratio"], self.PRESSz0, self.THETA
        )[0]
        self.moistlayer = moistlayer

        ### constants ###
        self.GRAVG = inputs["G"]
        self.CP_DRY = inputs["CP_DRY"]
        self.RGAS_DRY = inputs["RGAS_DRY"]
        self.RGAS_V = inputs["RGAS_V"]
        self.RC_DRY = self.RGAS_DRY / self.CP_DRY
        self.RCONST = 1 + self.qvapz0 * self.RGAS_V / self.RGAS_DRY
        self.P1000 = 100000  # P_1000 = 1000 hPa [Pa]
        self.CP0 = inputs["CP0"]
        self.RHO0 = inputs["RHO0"]
        self.Mr_ratio = inputs["Mr_ratio"]

        alpha = PRESSz0 / (self.RCONST * self.P1000)
        TEMPz0 = THETA * np.power(alpha, self.RC_DRY)  # temperature at z=0m [K]
        beta = (1 + self.qvapz0) / self.RCONST / self.RGAS_DRY
        self.RHOz0 = beta * self.PRESSz0 / TEMPz0

    def hydrostatic_adiabatic_profile(self, ZCOORDS):
        """returns *profile* of density (not the density itself!)
        rho = rhoprofile^((1-RC_DRY)/RC_DRY) = profile^pow"""
        pow = 1 / self.RC_DRY - 1

        Aa = (1 + self.qvapz0) * np.power(self.P1000, self.RC_DRY)
        Aa = self.THETA * self.RGAS_DRY / Aa
        Aconst = self.RCONST * np.power(Aa, (1 / (1 - self.RC_DRY)))

        RHOconst = -1 * self.GRAVG * self.RC_DRY / Aconst
        RHOprofile = np.power(self.RHOz0, 1 / pow) + RHOconst * ZCOORDS  # RHO^pow

        return RHOprofile, Aconst

    def hydrostatic_adiabatic_thermo(self, ZCOORDS):
        RHOprof, Aconst = self.hydrostatic_adiabatic_profile(ZCOORDS)
        # RHO = np.power(RHOprof, (1 / self.RC_DRY - 1))

        PRESS = Aconst * np.power(RHOprof, 1 / self.RC_DRY)

        TEMPconst = np.power(Aconst / (self.RCONST * self.P1000), self.RC_DRY)
        TEMP = self.THETA * TEMPconst * RHOprof

        return PRESS, TEMP

    def rhotilda(self, ZCOORDS):
        """returns dimensionless rho_dry profile for use in stream function"""
        PRESS, TEMP = self.hydrostatic_adiabatic_thermo(ZCOORDS)

        RHO_DRY = PRESS / ((self.RGAS_DRY + self.qvapz0 * self.RGAS_V) * TEMP)

        rhotilda = RHO_DRY / self.RHO0

        return rhotilda

    def generate_qvap(self, zfulls, xfulls, PRESS, TEMP):
        qvaps = qparams_to_qvap(
            self.qvapmethod, self.qvapparams, self.Mr_ratio, PRESS, TEMP
        )
        qvap = np.where(zfulls < self.Zbase, qvaps[0], qvaps[1])

        if self.moistlayer:
            z1, z2 = self.moistlayer["z1"], self.moistlayer["z2"]
            x1, x2 = self.moistlayer["x1"], self.moistlayer["x2"]
            mlqvap = sratio2qvap(
                self.moistlayer["mlsratio"], PRESS, TEMP, self.Mr_ratio
            )
            moistregion = (
                (zfulls >= z1) & (zfulls < z2) & (xfulls >= x1) & (xfulls < x2)
            )
            qvap = np.where(moistregion, mlqvap, qvap)

        return qvap

    def create_default_windsgen(self, WMAX, Zlength, Xlength, VVEL):
        return DryAdiabat2DFlowField(WMAX, Zlength, Xlength, VVEL, self)

    def generate_thermo(self, gbxbounds, ndims, ntime):
        zfulls, xfulls, yfulls = rgrid.fullcoords_forallgridboxes(gbxbounds, ndims)
        PRESS, TEMP = self.hydrostatic_adiabatic_thermo(zfulls)

        qvap = self.generate_qvap(zfulls, xfulls, PRESS, TEMP)

        shape_cen = int(
            ntime * np.prod(ndims)
        )  # = no. data for var defined at gridbox centers
        THERMODATA = {
            "PRESS": np.tile(PRESS, ntime),
            "TEMP": np.tile(TEMP, ntime),
            "qvap": np.tile(qvap, ntime),
            "qcond": np.full(shape_cen, self.qcond),
        }

        return THERMODATA


class HydrostaticLapseRates:
    """create thermodynamics that's constant in time and in hydrostatic equillibrium and
    following temperature and qvap (adiabats) with constant lapse rates above/below zbase.
    Qcond is uniform and constant."""

    def __init__(
        self,
        config_filename,
        constants_filename,
        PRESS0,
        TEMP0,
        qvap0,
        Zbase,
        TEMPlapses,
        qvaplapses,
        qcond,
    ):
        self.PRESS0 = PRESS0  # surface pressure [Pa]
        self.TEMP0 = TEMP0  # surface temperature [T]
        self.qvap0 = qvap0  # surface water vapour content [Kg/Kg]
        self.Zbase = Zbase  # cloud base height [m]
        self.TEMPlapses = TEMPlapses  # temp lapse rates [below, above] Zbase [K km^-1]
        self.qvaplapses = (
            qvaplapses  # qvap lapse rates [below, above] Zbase [g/Kg km^-1]
        )

        self.qcond = qcond  # liquid water content [Kg/Kg]

        inputs = thermoinputsdict(config_filename, constants_filename)
        self.GRAVG = inputs["G"]
        self.RGAS_DRY = inputs["RGAS_DRY"]
        self.Mr_ratio = inputs["Mr_ratio"]

    def temp1(self, z):
        """note unit conversion of input lapse rates:
        templapse rate = -dT/dz [K km^-1]  -->  [K m^-1]"""
        temp1 = self.TEMP0 - self.TEMPlapses[0] / 1000 * z
        if np.any((temp1 <= 0.0)):
            raise ValueError("TEMP > 0.0K")
        return temp1

    def temp2(self, z):
        """note unit conversion of input lapse rates:
        templapse rate = -dT/dz [K km^-1]  -->  [K m^-1]"""
        T_Zbase = self.temp1(self.Zbase)  # TEMP at Zbase
        temp2 = T_Zbase - self.TEMPlapses[1] / 1000 * (z - self.Zbase)
        if np.any((temp2 <= 0.0)):
            raise ValueError("TEMP > 0.0K")
        return temp2

    def hydrostatic_pressure(self, P0, integral):
        exponent = -self.GRAVG / self.RGAS_DRY * integral
        return P0 * np.exp(exponent)

    def press1(self, z):
        """hydrostatic pressure for value z where z <= self.Zbase"""
        P0 = self.PRESS0
        integral = integrate.quad(lambda x: 1 / self.temp1(x), 0.0, z)[0]
        return self.hydrostatic_pressure(P0, integral)

    def press2(self, z):
        """hydrostatic pressure for value z where z > self.Zbase"""
        P0 = self.press1(self.Zbase)
        integral = integrate.quad(lambda x: 1 / self.temp2(x), self.Zbase, z)[0]
        return self.hydrostatic_pressure(P0, integral)

    def qvap1(self, z):
        """note unit conversion of input lapse rates:
        qvaplapse rate = -dqvap/dz [g/Kg km^-1]  -->  [m^-1]"""

        if self.qvaplapses[0] == "saturated":
            sratio = 1.001
            qvap1 = sratio2qvap(sratio, self.press2(z), self.temp2(z), self.Mr_ratio)
        else:
            qvap1 = self.qvap0 - self.qvaplapses[0] / 1e6 * z

        if np.any((qvap1 <= 0.0)):
            raise ValueError("TEMP > 0.0K")
        return qvap1

    def qvap2(self, z):
        """note unit conversion of input lapse rates:
        qvaplapse rate = -dqvap/dz [g/Kg km^-1]  -->  [m^-1]"""
        if self.qvaplapses[1] == "saturated":
            sratio = 1.001
            qvap2 = sratio2qvap(sratio, self.press2(z), self.temp2(z), self.Mr_ratio)
        else:
            qvap_Zbase = self.qvap1(self.Zbase)  # qvap at Zbase
            qvap2 = qvap_Zbase - self.qvaplapses[1] / 1e6 * (z - self.Zbase)

        if np.any((qvap2 <= 0.0)):
            raise ValueError("TEMP > 0.0K")
        return qvap2

    def below_above_zbase(self, zfulls, func1, func2):
        return np.where(zfulls <= self.Zbase, func1(zfulls), func2(zfulls))

    def below_above_zbase_qvap(self, zfulls):
        qvap = []
        for z in zfulls:
            if z < self.Zbase:
                qvap.append(self.qvap1(z))
            else:
                qvap.append(self.qvap2(z))

        return np.asarray(qvap)

    def below_above_zbase_pressure(self, zfulls):
        PRESS = []
        for z in zfulls:
            if z < self.Zbase:
                PRESS.append(self.press1(z))
            else:
                PRESS.append(self.press2(z))

        return np.asarray(PRESS)

    def hydrostatic_lapserates_thermo(self, zfulls):
        TEMP = self.below_above_zbase(zfulls, self.temp1, self.temp2)
        PRESS = self.below_above_zbase_pressure(zfulls)
        qvap = self.below_above_zbase_qvap(zfulls)

        return TEMP, PRESS, qvap

    def generate_thermo(self, gbxbounds, ndims, ntime):
        zfulls, xfulls, yfulls = rgrid.fullcoords_forallgridboxes(gbxbounds, ndims)

        TEMP, PRESS, qvap = self.hydrostatic_lapserates_thermo(zfulls)

        shape_cen = int(
            ntime * np.prod(ndims)
        )  # = no. data for var defined at gridbox centers
        THERMODATA = {
            "PRESS": np.tile(PRESS, ntime),
            "TEMP": np.tile(TEMP, ntime),
            "qvap": np.tile(qvap, ntime),
            "qcond": np.full(shape_cen, self.qcond),
        }

        return THERMODATA


def linear_func(
    x: Union[np.ndarray, xr.DataArray], f_0: float = 0, slope: float = 1
) -> Union[np.ndarray, xr.DataArray]:
    """
    Linear function.

    :math:`y = slope * x + f_0`

    Parameters
    ----------
    x : np.ndarray
        The input array
    f_0 : float, optional
        The y-intercept, by default 0
    slope : float, optional
        The slope of the function, by default 1

    Returns
    -------
    np.ndarray
        The output array of the linear function

    """
    return slope * x + f_0


def split_linear_func(
    x: Union[np.ndarray, xr.DataArray],
    f_0: float = 2,
    slope_1: float = 1,
    slope_2: float = 2,
    x_split: float = 800,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Split the array x into two arrays at the point x_split. The function is the
    concatenation of two linear functions with different slopes.

    :math:`y_1 = slope_1 * x + f_0` for x <= x_split
    :math:`y_2 = slope_2 * x + f_0 + (slope_1 - slope_2) * x_split` for x > x_split

    Parameters
    ----------
    x : np.ndarray
        The input array
    f_0 : float, optional
        The y-intercept, by default 2
    slope_1 : float, optional
        The slope of the first linear function, by default 1
    slope_2 : float, optional
        The slope of the second linear function, by default 2
    x_split : float, optional
        The x value at which the array is split, by default 800

    Returns
    -------
    np.ndarray
        The sum of the two linear functions

    Examples
    --------
    >>> x = np.arange(0, 1000, 100)
    >>> split_linear(x, f_0=2, slope_1=1, slope_2=2, x_split=800)
    array([  2., 102., 202., 302., 402., 502., 602., 702., 802., 902.])
    """

    if isinstance(x, np.ndarray):
        x_1 = np.where(x <= x_split, x, np.nan)
        x_2 = np.where(x > x_split, x, np.nan)

        y_1 = linear_func(x=x_1, f_0=f_0, slope=slope_1)
        y_2 = linear_func(x=x_2, f_0=f_0 + (slope_1 - slope_2) * x_split, slope=slope_2)

        y = np.where(x <= x_split, y_1, y_2)
        return y
    elif isinstance(x, xr.DataArray):
        x_1 = x.where(x <= x_split)
        x_2 = x.where(x > x_split)

        y_1 = linear_func(x=x_1, f_0=f_0, slope=slope_1)
        y_2 = linear_func(x=x_2, f_0=f_0 + (slope_1 - slope_2) * x_split, slope=slope_2)

        y = xr.where(x <= x_split, y_1, y_2)
        return y


class SplittedLapseRates:
    """
    Create thermodynamics that are constant in time.
    The thermodynamical properties follow a Splitted Linear Function which
    changes slopes at the `cloud_base_height`.
    From the splitted linear functions of
    - potential temperature,
    - relative humidity,
    - pressure,
    the thermodynamic profiles are generated.

    The return profiles are the
    - ambient temperature,
    - specific humidity and
    - vertical velocity
    profiles, which consist of two slopes.

    It follows the `split_linear_function`:
    - :math:`y_1 = slope_1 * x + f_0` for x <= x_split
    - :math:`y_2 = slope_2 * x + f_0 + (slope_1 - slope_2) * x_split` for x > x_split

    The winds are to be considered constant in time and uniform throughout the domain.
    If the `Wlength` is greater than 0.0, the vertical velocity profile is a sinusoidal profile
    with amplitude `WMAX` and wavelength `2*Wlength`.


    Parameters
    ----------
    config_filename : str
        Path to the configuration file.
    constants_filename : str
        Path to the constants file.
    cloud_base_height : float
        Height of the cloud base.
    pressure_0 : float
        Initial pressure at the surface.
    potential_temperature_0 : float
        Initial potential temperature at the surface.
    relative_humidity_0 : float
        Initial relative humidity at the surface.
    pressure_lapse_rates : Union[np.ndarray, Tuple[float, float]]
        Tuple of pressure lapse rates below and above the cloud base.
        Units of Pa/m.
    potential_temperature_lapse_rates : Union[np.ndarray, Tuple[float, float]]
        Tuple of Potential temperature lapse rates below and above the cloud base.
        Units of K/m.
    relative_humidity_lapse_rates : Union[np.ndarray, Tuple[float, float]]
        Tuple of relative humidity lapse rates below and above the cloud base.
        Units of 1/m. NOT in percentage / m.
    w_maximum : float
        Maximum vertical velocity.
    u_velocity : float
        U component of the wind velocity.
    v_velocity : float
        V component of the wind velocity.
    Wlength : float
        Wavelength for the vertical velocity profile.

    Attributes:
        cloud_base_height (float): Height of the cloud base.
        pressure_0 (float): Initial pressure.
        potential_temperature_0 (float): Initial potential temperature.
        relative_humidity_0 (float): Initial relative humidity.
        pressure_lapse_rates (Union[np.ndarray, Tuple[float, float]]): Pressure lapse rates.
        potential_temperature_lapse_rates (Union[np.ndarray, Tuple[float, float]]): Potential temperature lapse rates.
        relative_humidity_lapse_rates (Union[np.ndarray, Tuple[float, float]]): Relative humidity lapse rates.
        w_maximum (float): Maximum vertical velocity.
        u_velocity (float): U component of the wind velocity.
        v_velocity (float): V component of the wind velocity.
        Wlength (float): Wavelength for the vertical velocity profile.
        Mr_ratio (float): Mixing ratio.
        GRAVG (float): Gravitational constant.
        RGAS_DRY (float): Gas constant for dry air.
    Methods:
        __init__(self, config_filename, constants_filename, cloud_base_height, pressure_0, potential_temperature_0, relative_humidity_0, pressure_lapse_rates, potential_temperature_lapse_rates, relative_humidity_lapse_rates, w_maximum, u_velocity, v_velocity, Wlength):
            Initializes the SplittedLapseRates class with the given parameters.
        pressure(self, z: np.ndarray) -> np.ndarray:
            Create the pressure profile from the given input parameters.
        potential_temperature(self, z: np.ndarray) -> np.ndarray:
            Create the potential temperature profile from the given input parameters.
        relative_humidity(self, z: np.ndarray) -> np.ndarray:
            Create the relative humidity profile from the given input parameters.
        temperature(self, z: np.ndarray) -> np.ndarray:
            Create the ambient temperature profile from the given input parameters.
        specific_humidity(self, z: np.ndarray) -> np.ndarray:
            Create the specific humidity profile from the given input parameters.
        wvel_profile(self, gbxbounds, ndims, ntime):
            Returns updraught (w always >=0.0) sinusoidal profile with amplitude WMAX and wavelength 2*Wlength.
        generate_winds(self, gbxbounds, ndims, ntime, THERMODATA):
            Generates wind profiles and updates the THERMODATA dictionary.
        generate_thermo(self, gbxbounds, ndims, ntime):
            Generates thermodynamic profiles and returns the THERMODATA dictionary.
            This is the main method to be called for the class.
    """

    def __init__(
        self,
        config_filename,
        constants_filename,
        cloud_base_height: float,
        pressure_0: float,
        potential_temperature_0: float,
        relative_humidity_0: float,
        pressure_lapse_rates: Union[np.ndarray, Tuple[float, float]],
        potential_temperature_lapse_rates: Union[np.ndarray, Tuple[float, float]],
        relative_humidity_lapse_rates: Union[np.ndarray, Tuple[float, float]],
        qcond: float,
        w_maximum: float,
        u_velocity: float,
        v_velocity: float,
        Wlength: float,
    ):
        self.cloud_base_height = cloud_base_height
        self.pressure_0 = pressure_0
        self.potential_temperature_0 = potential_temperature_0
        self.relative_humidity_0 = relative_humidity_0

        if any(
            (
                np.size(pressure_lapse_rates) != 2,
                np.size(potential_temperature_lapse_rates) != 2,
                np.size(relative_humidity_lapse_rates) != 2,
            )
        ):
            raise ValueError("The lapse rates need to be of size 2")

        self.pressure_lapse_rates = pressure_lapse_rates
        self.potential_temperature_lapse_rates = potential_temperature_lapse_rates
        self.relative_humidity_lapse_rates = relative_humidity_lapse_rates
        self.qcond = qcond
        self.w_maximum = w_maximum
        self.u_velocity = u_velocity
        self.v_velocity = v_velocity
        self.Wlength = Wlength

        inputs = thermoinputsdict(config_filename, constants_filename)
        self.Mr_ratio = inputs["Mr_ratio"]
        self.GRAVG = inputs["G"]
        self.RGAS_DRY = inputs["RGAS_DRY"]
        self.Mr_ratio = inputs["Mr_ratio"]

    def pressure(
        self, z: Union[np.ndarray, xr.DataArray]
    ) -> Union[np.ndarray, xr.DataArray]:
        """
        Crete the potential temperature profile from the given input parameters
        given in the __init__ of the class.

        It uses the ``split_linear_func`` function
        """

        return split_linear_func(
            x=z,
            f_0=self.pressure_0,
            slope_1=self.pressure_lapse_rates[0],
            slope_2=self.pressure_lapse_rates[1],
            x_split=self.cloud_base_height,
        )

    def potential_temperature(
        self, z: Union[np.ndarray, xr.DataArray]
    ) -> Union[np.ndarray, xr.DataArray]:
        """
        Crete the potential temperature profile from the given input parameters
        given in the __init__ of the class.

        It uses the ``split_linear_func`` function
        """

        return split_linear_func(
            x=z,
            f_0=self.potential_temperature_0,
            slope_1=self.potential_temperature_lapse_rates[0],
            slope_2=self.potential_temperature_lapse_rates[1],
            x_split=self.cloud_base_height,
        )

    def relative_humidity(
        self, z: Union[np.ndarray, xr.DataArray]
    ) -> Union[np.ndarray, xr.DataArray]:
        """
        Crete the relative humidity profile from the given input parameters
        given in the __init__ of the class.

        It uses the ``split_linear_func`` function
        """

        return split_linear_func(
            x=z,
            f_0=self.relative_humidity_0,
            slope_1=self.relative_humidity_lapse_rates[0],
            slope_2=self.relative_humidity_lapse_rates[1],
            x_split=self.cloud_base_height,
        )

    def temperature(
        self, z: Union[np.ndarray, xr.DataArray]
    ) -> Union[np.ndarray, xr.DataArray]:
        """
        Crete the ambient temperature profile from the given input parameters
        given in the __init__ of the class.

        The ambient temperature computed using the
        - potential temperature
        - pressure
        """

        return temperature_from_potential_temperature_pressure(
            potential_temperature=self.potential_temperature(z=z),
            pressure=self.pressure(z=z),
            pressure_reference=self.pressure_0,
        )

    def specific_humidity(
        self, z: Union[np.ndarray, xr.DataArray]
    ) -> Union[np.ndarray, xr.DataArray]:
        """
        Crete the specific humidity profile from the given input parameters
        given in the __init__ of the class.

        It uses the ``split_linear_func`` function.
        The specific humidity is directly computed using the relative humditiy
        """

        # the relative humdity here is given dimensionless
        # for the specific humidity calculations, we need relative humidity to be
        # in percentage !!!

        return specific_humidity_from_relative_humidity_temperature_pressure(
            relative_humidity=self.relative_humidity(z=z),
            temperature=self.temperature(z=z),
            pressure=self.pressure(z=z),
        )

    def wvel_profile(self, gbxbounds, ndims, ntime):
        """returns updraught (w always >=0.0) sinusoidal
        profile with amplitude WMAX and wavelength 2*Wlength"""

        zfaces = rgrid.coords_forgridboxfaces(gbxbounds, ndims, "z")[0]
        WVEL = self.w_maximum * np.sin(np.pi * zfaces / (2 * self.Wlength))

        WVEL[WVEL < 0.0] = 0.0

        return np.tile(WVEL, ntime)

    def generate_winds(self, gbxbounds, ndims, ntime, THERMODATA):
        THERMODATA = constant_winds(
            ndims=ndims,
            ntime=ntime,
            THERMODATA=THERMODATA,
            WVEL=self.w_maximum,
            UVEL=self.u_velocity,
            VVEL=self.v_velocity,
        )
        if self.Wlength > 0.0:
            THERMODATA["WVEL"] = self.wvel_profile(gbxbounds, ndims, ntime)

        return THERMODATA

    def generate_thermo(self, gbxbounds, ndims, ntime):
        zfulls, xfulls, yfulls = rgrid.fullcoords_forallgridboxes(gbxbounds, ndims)

        temperature = self.temperature(zfulls)
        pressure = self.pressure(zfulls)
        specific_humidity = self.specific_humidity(zfulls)

        shape_cen = int(ntime * np.prod(ndims))
        THERMODATA = {
            "PRESS": np.tile(pressure, ntime),
            "TEMP": np.tile(temperature, ntime),
            "qvap": np.tile(specific_humidity, ntime),
            "qcond": np.full(shape_cen, self.qcond),
        }

        THERMODATA = self.generate_winds(gbxbounds, ndims, ntime, THERMODATA)

        return THERMODATA
