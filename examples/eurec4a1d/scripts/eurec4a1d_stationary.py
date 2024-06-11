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
import os
import sys
import shutil
import math
import numpy as np
import yaml
from pathlib import Path
from io import StringIO

print(f"Enviroment: {sys.prefix}")

path2CLEO = Path(sys.argv[1])
path2build = Path(sys.argv[2])
path2home = Path(sys.argv[3])
origin_config_file = Path(sys.argv[4])
origin_cloud_observation_file = Path(sys.argv[5])
raw_dir = Path(sys.argv[6])

if "sdm_pysd_env312" not in sys.prefix:
    sys.path.append(str(path2CLEO))  # for imports from pySD package
    sys.path.append(
        str(path2CLEO / "examples/exampleplotting/")
    )  # for imports from example plotting package

# %%

from pySD import editconfigfile
from pySD.gbxboundariesbinary_src import read_gbxboundaries as rgrid
from pySD.gbxboundariesbinary_src import create_gbxboundaries as cgrid
from pySD.initsuperdropsbinary_src import (
    crdgens,
    probdists,
    rgens,
    attrsgen,
)
from pySD.initsuperdropsbinary_src import create_initsuperdrops as csupers
from pySD.initsuperdropsbinary_src import read_initsuperdrops as rsupers
from pySD.thermobinary_src import thermogen
from pySD.thermobinary_src import create_thermodynamics as cthermo
from pySD.thermobinary_src import read_thermodynamics as rthermo


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


# %%
### ---------------------------------------------------------------- ###
### ----------------------- INPUT PARAMETERS ----------------------- ###
### ---------------------------------------------------------------- ###
### --- essential paths and filenames --- ###
# path and filenames for creating initial SD conditions

constants_file = path2CLEO / "libs/cleoconstants.hpp"
bin_path = path2build / "bin/"
share_path = path2build / "share/"

raw_dir.mkdir(exist_ok=True, parents=True)

# create the raw directory for the individual cloud observation.
# Here the output data will be stored.

# CREATE A CONFIG FILE TO BE UPDATED
with open(origin_config_file, "r") as f:
    eurec4a1d_config = yaml.safe_load(f)


with open(origin_cloud_observation_file, "r") as f:
    cloud_observation_config = yaml.safe_load(f)


identification_type = cloud_observation_config["cloud"]["identification_type"]
cloud_id = cloud_observation_config["cloud"]["cloud_id"]

raw_individual_dir = raw_dir / f"{identification_type}_{cloud_id}"
raw_individual_dir.mkdir(exist_ok=True)

config_dir = raw_individual_dir / "config"
config_dir.mkdir(exist_ok=True)

# copy the cloud config file to the raw directory and use it
config_file = config_dir / "eurec4a1d_config.yaml"
cloud_observation_file = config_dir / "cloud_config.yaml"

shutil.copy(origin_config_file, config_file)
shutil.copy(origin_cloud_observation_file, cloud_observation_file)


share_path_individual = raw_individual_dir / "share"
share_path_individual.mkdir(exist_ok=True)


# path and file names for output data
eurec4a1d_config["outputdata"].update(
    setup_filename=str(config_dir / "eurec4a1d_setup.txt"),
    stats_filename=str(config_dir / "eurec4a1d_stats.txt"),
    zarrbasedir=str(raw_individual_dir / "eurec4a1d_sol.zarr"),
)

setup_file = eurec4a1d_config["outputdata"]["setup_filename"]
dataset_file = eurec4a1d_config["outputdata"]["zarrbasedir"]

eurec4a1d_config["coupled_dynamics"].update(
    **dict(
        type="fromfile",  # type of coupled dynamics to configure
        press=str(
            share_path_individual / "eurec4a1d_dimlessthermo_press.dat"
        ),  # binary filename for pressure
        temp=str(
            share_path_individual / "eurec4a1d_dimlessthermo_temp.dat"
        ),  # binary filename for temperature
        qvap=str(
            share_path_individual / "eurec4a1d_dimlessthermo_qvap.dat"
        ),  # binary filename for vapour mixing ratio
        qcond=str(
            share_path_individual / "eurec4a1d_dimlessthermo_qcond.dat"
        ),  # binary filename for liquid mixing ratio
        wvel=str(
            share_path_individual / "eurec4a1d_dimlessthermo_wvel.dat"
        ),  # binary filename for vertical (coord3) velocity
        thermo=str(
            share_path_individual / "eurec4a1d_dimlessthermo.dat"
        ),  # binary filename for thermodynamic profiles
    )
)

eurec4a1d_config["inputfiles"].update(
    grid_filename=str(
        share_path_individual / "eurec4a1d_ddimlessGBxboundaries.dat"
    ),  # binary filename for initialisation of GBxs / GbxMaps
    constants_filename=str(
        path2CLEO / "libs/cleoconstants.hpp"
    ),  # filename for constants
)
eurec4a1d_config["initsupers"].update(
    initsupers_filename=str(share_path_individual / "eurec4a1d_dimlessSDsinit.dat")
)

grid_file = eurec4a1d_config["inputfiles"]["grid_filename"]
init_superdroplets_file = eurec4a1d_config["initsupers"]["initsupers_filename"]
thermodynamics_file = eurec4a1d_config["coupled_dynamics"]["thermo"]


# %%
### ---------------------------------------------------------------- ###
### --- SETTING UP THERMODYNAMICS AND SUPERDROPLET INITIAL SETUP --- ###
### ---------------------------------------------------------------- ###


### --- settings for 1-D gridbox boundaries --- ###
cloud_altitude = cloud_observation_config["cloud"]["altitude"][0]
# only use integer precision
cloud_altitude = int(cloud_altitude)

cloud_thickness = 100

cloud_bottom = cloud_altitude - cloud_thickness / 2
cloud_top = cloud_altitude + cloud_thickness / 2
vertical_resolution = 20

# zgrid       = [0, cloud_top, vertical_resolution]      # evenly spaced zhalf coords [zmin, zmax, zdelta] [m]
# zgrid contains the boundaries of the gridboxes
# make sure to include the cloud bottom
zgrid = np.arange(0, cloud_bottom + vertical_resolution, vertical_resolution)
# add the cloud top as the upper boundary for the top gridbox
zgrid = np.append(zgrid, cloud_top)

xgrid = np.array([0, 20])  # array of xhalf coords [m]
ygrid = np.array([0, 20])  # array of yhalf coords [m]

air_temperature_params = cloud_observation_config["thermodynamics"]["air_temperature"][
    "parameters"
]
specific_humidity_params = cloud_observation_config["thermodynamics"][
    "specific_humidity"
]["parameters"]

### --- settings for 1-D Thermodynamics --- ###
pressure_bottom = 101315  # [Pa]
temperature_bottom = air_temperature_params["f_0"][0]  # [K]
temperature_lapse_rate = (
    np.array(air_temperature_params["slopes"]) * -1e3
)  # -1e3 due to conversion from dT/dz [K/m] to -dT/dz [K/km]
specific_humidity_bottom = specific_humidity_params["f_0"][0]  # [Kg/Kg]
specific_humidity_lapse_rate = (
    np.array(specific_humidity_params["slopes"]) * -1e6
)  # -1e6 due to conversion from dvap/dz [kg/kg m^-1] to -dvap/dz [g/Kg km^-1]
liquid_water_content = 0.0  # [Kg/Kg]
w_maximum = 0.0  # [m/s]
w_length = (
    0  # [m] use constant W (w_length=0.0), or sinusoidal 1-D profile below cloud base
)

z_split_temp = air_temperature_params["x_split"]  # [m]
z_split_qvap = specific_humidity_params["x_split"]  # [m]

# create the base of the cloud as the mean of the two splits
boundary_superdroplet_spawning = int(cloud_bottom - vertical_resolution)  # [m]

### --- settings for initial superdroplets --- ###
sd_per_gridbox = eurec4a1d_config["initsupers"][
    "initnsupers"
]  # number of superdroplets per gridbox

# initial superdroplet radii (and implicitly solute masses)
radius_minimum = 1e-7
radius_maximum = 1e-3
radius_span = [
    radius_minimum,
    radius_maximum,
]  # min and max range of radii to sample [m]

# initial superdroplet attributes
psd_params = cloud_observation_config["particle_size_distribution"]["parameters"]

# settings for initial superdroplet multiplicies with ATR and Aerosol from Lohmann et. al 2016 Fig. 5.5
geometric_means = psd_params["geometric_means"]
geometric_sigmas = psd_params["geometric_sigmas"]
scale_factors = psd_params["scale_factors"]
number_concentration = np.sum(scale_factors)

### ---------------------------------------------------------------- ###
### UPDATE THE BOUNDARY CONDITIONS FOR THE CONFIG FILE ###
### ---------------------------------------------------------------- ###

eurec4a1d_config["boundary_conditions"].update(
    COORD3LIM=boundary_superdroplet_spawning,  # SDs added to domain with coord3 >= boundary_superdroplet_spawning [m]
    newnsupers=sd_per_gridbox,  # number of new super-droplets per gridbox
    MINRADIUS=radius_minimum,  # minimum radius of new super-droplets [m]
    MAXRADIUS=radius_maximum,  # maximum radius of new super-droplets [m]
    NUMCONC_a=scale_factors[0],  # number conc. of 1st droplet lognormal dist [m^-3]
    GEOMEAN_a=geometric_means[0],  # geometric mean radius of 1st lognormal dist [m]
    geosigma_a=geometric_sigmas[
        0
    ],  # geometric standard deviation of 1st lognormal dist
    NUMCONC_b=scale_factors[1],  # number conc. of 2nd droplet lognormal dist [m^-3]
    GEOMEAN_b=geometric_means[1],  # geometric mean radius of 2nd lognormal dist [m]
    geosigma_b=geometric_sigmas[
        1
    ],  # geometric standard deviation of 2nd lognormal dist
)


### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
# %%
### ---------------------------------------------------------------- ###
### ----- MODIFY THE CONFIG FILE PRIOR TO CREATING INPUT FILES ----- ###
### ---------------------------------------------------------------- ###

editconfigfile.edit_config_params(str(config_file), eurec4a1d_config)

### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###

# %%


### ---------------------------------------------------------------- ###
### ------------------- BINARY FILES GENERATION--------------------- ###
### ---------------------------------------------------------------- ###
### --- ensure build, share and bin directories exist --- ###
if path2CLEO == path2build:
    raise ValueError("build directory cannot be CLEO")
else:
    Path(path2build).mkdir(exist_ok=True)
    Path(share_path).mkdir(exist_ok=True)
    Path(bin_path).mkdir(exist_ok=True)
os.system("rm " + grid_file)
os.system("rm " + init_superdroplets_file)
os.system("rm " + thermodynamics_file[:-4] + "*")


### ----- write gridbox boundaries binary ----- ###
cgrid.write_gridboxboundaries_binary(grid_file, zgrid, xgrid, ygrid, constants_file)
with Capturing() as grid_info:
    rgrid.print_domain_info(constants_file, grid_file)

for line in grid_info:
    print(line)
    if "domain no. gridboxes:" in line:
        grid_dimensions = np.array(
            line.split(":")[-1].replace(" ", "").split("x"), dtype=int
        )
        total_number_gridboxes = int(np.prod(grid_dimensions))
        break

# Update the max number of superdroplets
renew_timesteps = (
    eurec4a1d_config["timesteps"]["T_END"]
    / eurec4a1d_config["timesteps"]["MOTIONTSTEP"]
)
# add 1000 to ensure enough space for new SDs
max_number_supers = int(math.ceil(renew_timesteps * sd_per_gridbox + 1000))
# get the total number of gridboxes
eurec4a1d_config["domain"].update(
    nspacedims=1, ngbxs=total_number_gridboxes, maxnsupers=max_number_supers
)

# update the config file
editconfigfile.edit_config_params(str(config_file), eurec4a1d_config)

# %%

### ----- write thermodynamics binaries ----- ###
thermodynamics_generator = thermogen.ConstHydrostaticLapseRates(
    configfile=config_file,
    constsfile=constants_file,
    PRESS0=pressure_bottom,
    TEMP0=temperature_bottom,
    qvap0=specific_humidity_bottom,
    Zbase=boundary_superdroplet_spawning,
    TEMPlapses=temperature_lapse_rate,
    qvaplapses=specific_humidity_lapse_rate,
    qcond=liquid_water_content,
    WMAX=w_maximum,
    UVEL=None,
    VVEL=None,
    Wlength=w_length,
)
cthermo.write_thermodynamics_binary(
    thermodynamics_file,
    thermodynamics_generator,
    config_file,
    constants_file,
    grid_file,
)

### ----- write initial superdroplets binary ----- ###
number_superdroplets = crdgens.nsupers_at_domain_top(
    grid_file, constants_file, sd_per_gridbox, boundary_superdroplet_spawning
)

# get total number of superdroplets
total_number_superdroplets = int(np.sum(list(number_superdroplets.values())))
eurec4a1d_config["initsupers"].update(initnsupers=total_number_superdroplets)
# update the config file
editconfigfile.edit_config_params(str(config_file), eurec4a1d_config)


# create initial superdroplets coordinates
corrd3_generator = crdgens.SampleCoordGen(True)  # sample coord3 randomly
corrd1_generator = None  # do not generate superdroplet coord2s
corrd2_generator = None  # do not generate superdroplet coord2s

# create initial superdroplets attributes
xi_prob_distribution = probdists.LnNormal(
    geometric_means, geometric_sigmas, scale_factors
)
radii_generator = rgens.SampleLog10RadiiGen(radius_span)

# create uniform dry radii
monodryr = 1e-9  # all SDs have this same dryradius [m]
dry_radii_generator = rgens.MonoAttrGen(monodryr)

# write initial superdroplets binary
initattrsgen = attrsgen.AttrsGenerator(
    radii_generator,
    dry_radii_generator,
    xi_prob_distribution,
    corrd3_generator,
    corrd1_generator,
    corrd2_generator,
)
csupers.write_initsuperdrops_binary(
    init_superdroplets_file,
    initattrsgen,
    config_file,
    constants_file,
    grid_file,
    number_superdroplets,
    number_concentration,
)
# ### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###

### ---------------------------------------------------------------- ###
### --------- MODIFY THE CONFIG FILE PRIOR TO RUNNING CLEO --------- ###
### ---------------------------------------------------------------- ###

editconfigfile.edit_config_params(str(config_file), eurec4a1d_config)

### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
### ---------------------- PLOT INIT FIGURES ----------------------- ###
### ---------------------------------------------------------------- ###

isfigures = [True, True]  # booleans for [making, saving] initialisation figures
figure_dir = str(raw_individual_dir / "figures") + "/"  # directory for saving figures

SDgbxs2plt = total_number_gridboxes - 1

### ----- show (and save) plots of binary file data ----- ###
if isfigures[0]:
    if isfigures[1]:
        Path(figure_dir).mkdir(exist_ok=True)
    rgrid.plot_gridboxboundaries(constants_file, grid_file, figure_dir, isfigures[1])
    rthermo.plot_thermodynamics(
        constants_file,
        config_file,
        grid_file,
        thermodynamics_file,
        figure_dir,
        isfigures[1],
    )
    rsupers.plot_initGBxs_distribs(
        config_file,
        constants_file,
        init_superdroplets_file,
        grid_file,
        figure_dir,
        isfigures[1],
        SDgbxs2plt,
    )
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###


# ### ---------------------------------------------------------------- ###
# ### ---------------------- RUN CLEO EXECUTABLE --------------------- ###
# ### ---------------------------------------------------------------- ###
# print("===============================")
# print("Running CLEO executable")
# os.chdir(path2build)
# os.system("pwd")
# os.system("rm -rf " + dataset_file)  # delete any existing dataset_file
# executable = str(path2build) + "/examples/eurec4a1d/src/eurec4a1D"
# print("Executable: " + executable)
# print("Config file: " + str(config_file))
# os.system(executable + " " + str(config_file))
# print("===============================")
# ### ---------------------------------------------------------------- ###
# ### ---------------------------------------------------------------- ###
