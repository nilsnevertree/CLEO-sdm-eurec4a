# %%
"""
----- CLEO -----
File: rainshaft1d.py
Project: rainshaft1d
Created Date: Friday 17th November 2023
Author: Clara Bayley (CB)
Additional Contributors:
-----
Last Modified: Wednesday 17th January 2024
Modified By: CB
-----
License: BSD 3-Clause "New" or "Revised" License
https://opensource.org/licenses/BSD-3-Clause
-----
Copyright (c) 2023 MPI-M, Clara Bayley
-----
File Description:
Script compiles and runs CLEO rain1D to create the
data and plots precipitation example given constant
1-D rainshaft thermodynamics read from a file
"""

# %%
import random
import re
import sys
from typing import Union, Tuple
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr

import sdm_eurec4a.input_processing.models as smodels
from sdm_eurec4a.conversions import msd_from_psd_dataarray

path2CLEO = Path(__file__).resolve().parents[3]

if "sdm_pysd_env312" not in sys.prefix:
    sys.path.append(str(path2CLEO))  # for imports from pySD package
    sys.path.append(
        str(path2CLEO / "examples/exampleplotting/")
    )  # for imports from example plotting package


from pySD.gbxboundariesbinary_src import create_gbxboundaries as cgrid
from pySD.gbxboundariesbinary_src import read_gbxboundaries as rgrid
from pySD.initsuperdropsbinary_src import (
    attrsgen,
    crdgens,
    probdists,
    rgens,
)
from pySD.initsuperdropsbinary_src import create_initsuperdrops as csupers
from pySD.initsuperdropsbinary_src import read_initsuperdrops as rsupers
from pySD.thermobinary_src import create_thermodynamics as cthermo
from pySD.thermobinary_src import read_thermodynamics as rthermo
from pySD.thermobinary_src import thermogen


print(f"Enviroment: {sys.prefix}")


path2build = path2CLEO / "builds_eurec4a1d"
path2home = path2CLEO.parents[1]
config_file_path = (
    path2CLEO / "examples/eurec4a1d/default_config/eurec4a1d_config_evolving.yaml"
)
constants_file_path = path2CLEO / "libs/cleoconstants.hpp"
binary_path = path2build / "bin/"
share_path = path2build / "share/"
cloud_observation_file_path = (
    path2CLEO / "examples/eurec4a1d/default_config/cloud_observation.yaml"
)
rawdirectory = path2CLEO / "data/output/raw/netcdf1"
rawdirectory.mkdir(exist_ok=True, parents=True)


class Capturing(list):
    """
    Context manager for capturing stdout from print statements.
    https://stackoverflow.com/a/16571630/16372843
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def parameters_dataset_to_dict(
    ds: xr.Dataset, mapping: Union[dict[str, str], Tuple[str]]
) -> dict:
    """
    Convert selected parameters from an xarray Dataset to a dictionary.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the parameters.
    mapping : Union[dict[str, str], Tuple[str]]
        A mapping of parameter names to extract from the Dataset. If a dictionary is provided,
        the keys are the new names for the parameters and the values are the names in the Dataset.
        If a tuple or list is provided, the parameter names are used as-is.

    Returns
    -------
    dict
        A dictionary where the keys are the parameter names (or new names if a dictionary was provided)
        and the values are the corresponding values from the Dataset.

    Raises
    ------
    TypeError
        If the mapping is not a dictionary or a tuple/list.
    """

    if isinstance(mapping, (list, tuple)):
        parameters = {key: float(ds[key].values) for key in mapping}
    elif isinstance(mapping, dict):
        parameters = {mapping[key]: float(ds[key].values) for key in mapping}
    else:
        raise TypeError("mapping must be a dict or a tuple")

    return parameters


def plot_11(ax: plt.Axes):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    g_min = min(x_min, y_min)
    g_max = max(x_max, y_max)
    g = (g_max, g_min)

    ax.plot(g, g, color="black", linestyle="--")


# data_dir = Path('/work/mh1126/m301096/softlinks/sdm-eurec4a/data/model/netcdf-parameters_v1.0')
data_dir = Path(
    r"C:\Users\Niebaum\Documents\Repositories\sdm-eurec4a\data\model\inputv4.0"
)

# Load the parameters datasets
psd_parameters = xr.open_dataset(data_dir / "particle_size_distribution_parameters.nc")
potential_temperature_parameters = xr.open_dataset(
    data_dir / "potential_temperature_parameters.nc"
)
relative_humidity_parameters = xr.open_dataset(
    data_dir / "relative_humidity_parameters.nc"
)
pressure_parameters = xr.open_dataset(data_dir / "pressure_parameters.nc")
# xr.testing.assert_allclose(potential_temperature_parameters['x_split'], relative_humidity_parameters['x_split'], rtol = 1e-12)

# %%


# create the individual raw directory
identification_type = "cluster"
cloud_id = np.random.choice(psd_parameters["cloud_id"])
raw_directory_individual_path = rawdirectory / f"{identification_type}_{cloud_id}"
raw_directory_individual_path.mkdir(exist_ok=True)


# CREATE A CONFIG FILE TO BE UPDATED
updated_configfile = dict()

# path and file names for output data
updated_configfile["outputdata"] = dict(
    setup_filename=str(raw_directory_individual_path / "eurec4a1d_setup.txt"),
    stats_filename=str(raw_directory_individual_path / "eurec4a1d_stats.txt"),
    zarrbasedir=str(raw_directory_individual_path / "eurec4a1d_sol.zarr"),
    maxchunk=2500000,
)

setup_filename = updated_configfile["outputdata"]["setup_filename"]
stats_filename = updated_configfile["outputdata"]["stats_filename"]
dataset_path = updated_configfile["outputdata"]["zarrbasedir"]


updated_configfile["coupled_dynamics"] = dict(
    type="fromfile",  # type of coupled dynamics to configure
    press=str(
        data_dir / "share/eurec4a1d_dimlessthermo_press.dat"
    ),  # binary filename for pressure
    temp=str(
        data_dir / "share/eurec4a1d_dimlessthermo_temp.dat"
    ),  # binary filename for temperature
    qvap=str(
        data_dir / "share/eurec4a1d_dimlessthermo_qvap.dat"
    ),  # binary filename for vapour mixing ratio
    qcond=str(
        data_dir / "share/eurec4a1d_dimlessthermo_qcond.dat"
    ),  # binary filename for liquid mixing ratio
    wvel=str(
        data_dir / "share/eurec4a1d_dimlessthermo_wvel.dat"
    ),  # binary filename for vertical (coord3) velocity
    thermo=str(
        data_dir / "share/eurec4a1d_dimlessthermo.dat"
    ),  # binary filename for thermodynamic profiles
)
updated_configfile["inputfiles"] = dict(
    constants_pathname="../libs/cleoconstants.hpp",  # name of file for values of physical constants
    gridfilename=str(
        data_dir / "share/eurec4a1d_ddimlessGBxboundaries.dat"
    ),  # binary filename for initialisation of GBxs / GbxMaps
)

updated_configfile["initsupers"] = dict(
    type="frombinary",
    initsupers_filename=str(data_dir / "share/eurec4a1d_dimlessSDsinit.dat"),
    initnsupers=0,  # Modify later!!!!
)

gridfile_path = updated_configfile["inputfiles"]["gridfilename"]
init_superdroplets_file_path = updated_configfile["initsupers"]["initsupers_filename"]
thermo_file_path = updated_configfile["coupled_dynamics"]["thermo"]

inputs_dict = csupers.initSDsinputsdict(config_file_path, constants_file_path)

# %%
### --- settings for 1-D gridbox boundaries --- ###
# only use integer precision
cloud_altitude = potential_temperature_parameters["x_split"].mean().values
cloud_altitude = int(cloud_altitude)

dz = 20
dz_cloud = 100
dx = 100
dy = 100

cloud_bottom = cloud_altitude - dz_cloud / 2

# below cloud
zgrid = np.arange(0, cloud_bottom + dz, dz)
# above cloud
zgrid_top = np.max(zgrid) + dz_cloud
zgrid = np.append(zgrid, zgrid_top)


gridboxvolume = dx * dy * dz

xgrid = np.array([0, dx])  # array of xhalf coords [m]
ygrid = np.array([0, dy])  # array of yhalf coords [m]


# create initial superdroplets coordinates
coord3gen = crdgens.SampleCoordGen(True)  # sample coord3 randomly
coord1gen = None  # do not generate superdroplet coord2s
coord2gen = None  # do not generate superdroplet coord2s

### ----- write gridbox boundaries binary ----- ###
cgrid.write_gridboxboundaries_binary(
    gridfile=gridfile_path,
    zgrid=zgrid,
    xgrid=xgrid,
    ygrid=ygrid,
    constsfile=constants_file_path,
)
with Capturing() as grid_info:
    rgrid.print_domain_info(constants_file_path, gridfile_path)

# extract the total number of gridboxes
for line in grid_info:
    print(line)
    if "domain no. gridboxes:" in line:
        grid_dimensions = np.array(
            line.split(":")[-1].replace(" ", "").split("x"), dtype=int
        )
        number_gridboxes_total = int(np.prod(grid_dimensions))
        break
# write total number of gridboxes to config file
updated_configfile["domain"] = dict(
    nspacedims=1,
    ngbxs=number_gridboxes_total,
)

# %%

### --- settings for initial superdroplets --- ###
number_superdroplets_per_gridbox = 512  # number of superdroplets per gridbox
number_superdroplets_total = number_superdroplets_per_gridbox
updated_configfile["initsupers"]["initnsupers"] = number_superdroplets_total

# initial superdroplet radii (and implicitly solute masses)
radius_span = slice(50e-6, 3e-3)  # min and max range of radii to sample [m]
radius_span_list = [radius_span.start, radius_span.stop]


# %%
# create fitted data

r = np.geomspace(radius_span.start, radius_span.stop, number_superdroplets_per_gridbox)
da_radius = xr.DataArray(r, coords=dict(radius=r))
da_bin_width = 0.5 * (da_radius.shift(radius=-1) - da_radius.shift(radius=1))
da_bin_width[0] = da_bin_width[1] - (da_bin_width[2] - da_bin_width[1])
da_bin_width[-1] = da_bin_width[-2] + (da_bin_width[-2] - da_bin_width[-3])


# the orginal form of the log normal distirbution is a double log normal distribution
# the t space is linear and the
# the mean and standard deviation are geometric
da_fitted_psd: xr.DataArray = smodels.double_ln_normal_distribution(  # type: ignore
    t=da_radius,
    mu1=psd_parameters["mu1"],
    mu2=psd_parameters["mu2"],
    scale_factor1=psd_parameters["scale_factor1"],
    scale_factor2=psd_parameters["scale_factor2"],
    sigma1=psd_parameters["sigma1"],
    sigma2=psd_parameters["sigma2"],
)

da_fitted_psd.attrs.update(
    long_name="Fitted particle size distribution",
    units="m^{-3} m^{-1} ",
    description="Fitted particle size distribution from the parameters",
)

da_fitted_msd = msd_from_psd_dataarray(da_fitted_psd * da_bin_width) / da_bin_width
da_fitted_msd.attrs.update(
    long_name="Fitted mass size distribution",
    units="kg m^{-3} m^{-1} ",
    description="Fitted mass size distribution from the parameters",
)

fitted_nbc = (da_fitted_psd * da_bin_width).sum("radius")
fitted_nbc.attrs.update(
    long_name="Fitted number concentration",
    units="m^{-3}",
    description="Fitted number concentration from the parameters",
)

fitted_lwc = (da_fitted_msd * da_bin_width).sum("radius")
fitted_lwc.attrs.update(
    long_name="Fitted liquid water content",
    units="kg m^{-3}",
    description="Fitted liquid water content from the parameters",
)

ds_fitted = xr.Dataset(
    dict(
        particle_size_distribution=da_fitted_psd,
        mass_size_distribution=da_fitted_msd,
        multiplicities=da_fitted_psd * gridboxvolume,
        number_concentration=fitted_nbc,
        liquid_water_content=fitted_lwc,
        bin_width=da_bin_width,
    )
)


# %%

# create initial superdroplets attributes
radii_generator = rgens.SampleLog10RadiiWithBinWidth(radius_span_list)
# create uniform dry radii
monodryr = 1e-13  # all SDs have this same dryradius [m]
dryradii_generator = rgens.MonoAttrGen(monodryr)

radii, bin_width = radii_generator(number_superdroplets_per_gridbox)
# %%

params1_linear_space = (
    smodels.GeometricMuSigmaScaleLog(
        geometric_mu_l=psd_parameters["mu1"],
        geometric_std_dev=psd_parameters["sigma1"],
        scale_l=psd_parameters["scale_factor1"],
    )
    .standardize()
    .get_geometric_parameters_linear(
        dict_keys=("geometric_mean1", "geometric_std_dev1", "scale_factor1")
    )
)

params2_linear_space = (
    smodels.GeometricMuSigmaScaleLog(
        geometric_mu_l=psd_parameters["mu2"],
        geometric_std_dev=psd_parameters["sigma2"],
        scale_l=psd_parameters["scale_factor2"],
    )
    .standardize()
    .get_geometric_parameters_linear(
        dict_keys=("geometric_mean2", "geometric_std_dev2", "scale_factor2")
    )
)

psd_parameters_linear_space = xr.Dataset(
    dict(**params1_linear_space, **params2_linear_space)
)

mapping = dict(
    geometric_mean1="geometric_mean1",
    geometric_mean2="geometric_mean2",
    geometric_std_dev1="geometric_std_dev1",
    geometric_std_dev2="geometric_std_dev2",
    scale_factor1="scale_factor1",
    scale_factor2="scale_factor2",
)

# %%

list_errors = []
list_cleo_psd = []
list_bin_width = []

pseudo_radii = da_radius


for cloud_id in tqdm.tqdm(psd_parameters["cloud_id"].values):
    parameters = parameters_dataset_to_dict(
        ds=psd_parameters_linear_space.sel(cloud_id=cloud_id),
        mapping=mapping,
    )

    xi_probability_distribution = probdists.DoubleLogNormal(**parameters)

    initial_attributes_generator = attrsgen.AttrsGeneratorBinWidth(
        radiigen=radii_generator,
        dryradiigen=dryradii_generator,
        xiprobdist=xi_probability_distribution,
        coord3gen=coord3gen,
        coord1gen=coord1gen,
        coord2gen=coord2gen,
    )

    try:
        radii, bin_width = radii_generator(number_superdroplets_per_gridbox)

        multiplicities = initial_attributes_generator.multiplicities(
            radii=radii,
            samplevol=gridboxvolume,
            bin_width=bin_width,
        )

        multiplicities = xr.DataArray(
            multiplicities, coords=[("radius", da_radius.values)]
        )
        multiplicities = multiplicities.expand_dims(cloud_id=[cloud_id])

        cleo_bin_width = xr.DataArray(bin_width, coords=[("radius", da_radius.values)])
        cleo_bin_width = cleo_bin_width.expand_dims(cloud_id=[cloud_id])

        list_cleo_psd.append(multiplicities)
        list_bin_width.append(cleo_bin_width)
        list_errors.append(0)

    except ValueError as ve:
        error_message = str(ve)
        match = re.search(r"ERROR, (\d+) out", error_message)
        if match:
            error_code = int(match.group(1))
            list_errors.append(error_code)
            print(f"Error code: {error_code}")
        else:
            raise ve

# %%
# da_errors = xr.DataArray(list_errors, coords=[psd_parameters["cloud_id"],])
ds_cleo = xr.Dataset(
    dict(
        multiplicities=xr.concat(list_cleo_psd, dim="cloud_id"),
        bin_width=xr.concat(list_bin_width, dim="cloud_id"),
    )
)
ds_cleo["multiplicities"].attrs.update(
    long_name="Multiplicity",
    units="m^{-3}",
    description="Number of superdroplets per unit volume and radius",
)
ds_cleo["bin_width"].attrs.update(
    long_name="Bin width",
    units="m",
    description="Width of the radius bin",
)
ds_cleo["particle_size_distribution"] = ds_cleo["multiplicities"] / gridboxvolume
ds_cleo["particle_size_distribution"].attrs.update(
    long_name="Particle size distribution",
    units="m^{-3}",
    description="Number of superdroplets per unit volume and radius",
)
ds_cleo["mass_size_distribution"] = msd_from_psd_dataarray(
    ds_cleo["particle_size_distribution"]
)
ds_cleo["number_concentration"] = (ds_cleo["particle_size_distribution"]).sum("radius")
ds_cleo["liquid_water_content"] = (ds_cleo["mass_size_distribution"]).sum("radius")


cloud_ids = ds_cleo["cloud_id"].values
# %%

fig, axs = plt.subplots(1, 3, figsize=(10, 5))

for _ax, variable in zip(
    axs, ["multiplicities", "particle_size_distribution", "mass_size_distribution"]
):
    # plot the multiplicity
    _ax.plot(
        ds_cleo["radius"],
        ds_cleo[variable].T,
        color="blue",
        alpha=0.2,
    )
    _ax.plot(
        ds_fitted["radius"],
        (ds_fitted[variable] * ds_fitted["bin_width"]).sel(cloud_id=cloud_ids),
        color="red",
        alpha=0.2,
    )
    _ax.set_xscale("log")
    _ax.set_xlabel("Radius [m]")


axs[0].set_ylabel("Multiplicity [1]")
axs[0].set_yscale("log")
axs[0].set_ylim(0.1, None)

axs[1].set_yscale("log")
axs[1].set_ylabel("Particle Size distirbution [1 m$^{-3}$]")

fig.tight_layout()

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(
    ds_fitted["number_concentration"].sel(cloud_id=cloud_ids),
    ds_cleo["number_concentration"],
)
axs[0].set_xlabel("Fitted NBC [m$^{-3}$]")
axs[0].set_ylabel("CLEO NBC [m$^{-3}$]")
plot_11(ax=axs[0])


axs[1].scatter(
    1e3 * ds_fitted["liquid_water_content"].sel(cloud_id=cloud_ids),
    1e3 * ds_cleo["liquid_water_content"],
)
axs[1].set_xlabel("Fitted LWC [g m$^{-3}$]")
axs[1].set_ylabel("CLEO LWC [g m$^{-3}$]")
plot_11(ax=axs[1])


# %%

# write last cloud to init SD file

csupers.write_initsuperdrops_binary(
    initsupersfile=init_superdroplets_file_path,
    initattrsgen=initial_attributes_generator,
    configfile=config_file_path,
    constsfile=constants_file_path,
    gridfile=gridfile_path,
    nsupers=number_superdroplets_total,
    NUMCONC=0,
)

# %%
# -------------------------
# --- Thermodynamics ------
# -------------------------

a = np.arange(1500)
altitude = xr.DataArray(a, coords=dict(altitude=a))

cloud_id = 81


slr = thermogen.SplittedLapseRates(
    configfile=config_file_path,
    constsfile=constants_file_path,
    cloud_base_height=relative_humidity_parameters["x_split"]
    .sel(cloud_id=cloud_id)
    .values,
    pressure_0=pressure_parameters["f_0"].sel(cloud_id=cloud_id).values,
    potential_temperature_0=potential_temperature_parameters["f_0"]
    .sel(cloud_id=cloud_id)
    .values,
    relative_humidity_0=relative_humidity_parameters["f_0"]
    .sel(cloud_id=cloud_id)
    .values,
    pressure_lapse_rates=(
        pressure_parameters["slope"].sel(cloud_id=cloud_id).values,
        pressure_parameters["slope"].sel(cloud_id=cloud_id).values,
    ),
    potential_temperature_lapse_rates=(
        potential_temperature_parameters["slope_1"].sel(cloud_id=cloud_id).values,
        potential_temperature_parameters["slope_2"].sel(cloud_id=cloud_id).values,
    ),
    relative_humidity_lapse_rates=(
        relative_humidity_parameters["slope_1"].sel(cloud_id=cloud_id).values,
        relative_humidity_parameters["slope_2"].sel(cloud_id=cloud_id).values,
    ),
    qcond=0.0,
    w_maximum=0.0,
    u_velocity=None,
    v_velocity=None,
    Wlength=0.0,
)

cthermo.write_thermodynamics_binary(
    thermo_file_path, slr, config_file_path, constants_file_path, gridfile_path
)


# %%

temperature = slr.temperature(altitude)
potential_temperature = slr.potential_temperature(altitude)
pressure = slr.pressure(altitude)
relative_humidity = slr.relative_humidity(altitude)
specific_humidity = slr.specific_humidity(altitude)

# %%

fig, ax = plt.subplots(1, 5, sharey=True, figsize=(15, 5))

ax[0].plot(pressure, altitude)
ax[0].set_ylabel("Pressure [Pa]")

ax[1].plot(temperature, altitude)
ax[1].set_ylabel("Temperature [K]")

ax[2].plot(potential_temperature, altitude)
ax[2].set_ylabel("Potential Temperature [K]")

ax[3].plot(specific_humidity, altitude)
ax[3].set_ylabel("Specific Humidity [kg/kg]")

ax[4].plot(relative_humidity, altitude)
ax[4].set_ylabel("Relative Humidity [%]")

fig.tight_layout()


# %%


# -------------------------
# USE ORIGINAL CLEO CODE FOR PLOTS
# -------------------------

inputs_dict = cthermo.thermoinputsdict(config_file_path, constants_file_path)

gbxbounds, ndims = rgrid.read_dimless_gbxboundaries_binary(
    gridfile_path, COORD0=inputs_dict["COORD0"], return_ndims=True, isprint=False
)
xyzhalf = rgrid.halfcoords_from_gbxbounds(gbxbounds, isprint=False)  # [m]
zhalf, xhalf, yhalf = [half / 1000 for half in xyzhalf]  # convery [m] to [km]
zfull, xfull, yfull = rgrid.fullcell_fromhalfcoords(zhalf, xhalf, yhalf)  # [m]

thermodata = rthermo.get_thermodynamics_from_thermofile(
    thermo_file_path, ndims, inputs=inputs_dict
)


isfigures = [True, True]  # booleans for [making, saving] initialisation figures
fig_dir = str(data_dir / "figures")
SDgbxs2plt = list(range(number_gridboxes_total - 2, number_gridboxes_total - 1))
SDgbxs2plt = [random.choice(SDgbxs2plt)]  # choose random gbx from list to plot


### ----- show (and save) plots of binary file data ----- ###
if isfigures[0]:
    rgrid.plot_gridboxboundaries(
        constants_file_path, gridfile_path, fig_dir, isfigures[1]
    )
    rthermo.plot_thermodynamics(
        constants_file_path,
        config_file_path,
        gridfile_path,
        thermo_file_path,
        fig_dir,
        isfigures[1],
    )
    rsupers.plot_initGBxs_distribs(
        configfile=config_file_path,
        constsfile=constants_file_path,
        initsupersfile=init_superdroplets_file_path,
        gridfile=gridfile_path,
        binpath=fig_dir,
        savefig=isfigures[1],
        gbxs2plt=SDgbxs2plt,
    )

# %%
number_superdroplets_per_gridbox = 100

dln_NUMCONC = probdists.DoubleLogNormal(**psd_parameters_linear_space)

cloud_id = 81

print(cloud_id)

parameters = parameters_dataset_to_dict(
    ds=psd_parameters_linear_space.sel(cloud_id=cloud_id), mapping=mapping
)

xi_probability_distribution = probdists.DoubleLogNormal(**parameters)

initial_attributes_generator = attrsgen.AttrsGeneratorBinWidth(
    radiigen=radii_generator,
    dryradiigen=dryradii_generator,
    xiprobdist=xi_probability_distribution,
    coord3gen=coord3gen,
    coord1gen=coord1gen,
    coord2gen=coord2gen,
)

radii, bin_width = radii_generator(number_superdroplets_per_gridbox)

multiplicities = initial_attributes_generator.multiplicities(
    radii=radii,
    samplevol=gridboxvolume,
    bin_width=bin_width,
)

NUMCONC = np.sum(multiplicities)


# multiplicities = xr.DataArray(
#     multiplicities,
#     coords=[("radius", radii)])
# multiplicities = multiplicities.expand_dims(cloud_id = [cloud_id])

cleo_bin_width = xr.DataArray(bin_width, coords=[("radius", radii)])
cleo_bin_width = cleo_bin_width.expand_dims(cloud_id=[cloud_id])

from pySD.gbxboundariesbinary_src import read_gbxboundaries as rbounds

gbxbounds = rbounds.read_dimless_gbxboundaries_binary(
    gridfile_path, COORD0=inputs_dict["COORD0"], isprint=False
)


multiplicities2, radii2, sol2 = initial_attributes_generator.generate_attributes(
    nsupers=number_superdroplets_per_gridbox,
    gridboxbounds=gbxbounds[0],
    RHO_SOL=inputs_dict["RHO_DRY"],
    NUMCONC=1,
)

psd1 = xr.DataArray(multiplicities, dims="radius")
psd2 = xr.DataArray(multiplicities2, dims="radius")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(radii, multiplicities)
axs[0].plot(radii2, multiplicities2)
plt.xscale("log")
plt.yscale("log")
# %%
