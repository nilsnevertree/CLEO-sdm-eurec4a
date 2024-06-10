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
configfile = Path(sys.argv[4])
cloud_observation_filepath = Path(sys.argv[5])
rawdirectory = Path(sys.argv[6])

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

constsfile = path2CLEO / "libs/cleoconstants.hpp"
binpath = path2build / "bin/"
sharepath = path2build / "share/"

rawdirectory.mkdir(exist_ok=True, parents=True)

# create the raw directory for the individual cloud observation.
# Here the output data will be stored.

# CREATE A CONFIG FILE TO BE UPDATED
with open(configfile, "r") as f:
    updated_configfile = yaml.safe_load(f)


with open(cloud_observation_filepath, "r") as f:
    cloud_observation_config = yaml.safe_load(f)


identification_type = cloud_observation_config["cloud"]["identification_type"]
cloud_id = cloud_observation_config["cloud"]["cloud_id"]

rawdirectory_individual = rawdirectory / f"{identification_type}_{cloud_id}"
rawdirectory_individual.mkdir(exist_ok=True)

config_dir = rawdirectory_individual / "config"
config_dir.mkdir(exist_ok=True)

# copy the cloud config file to the raw directory and use it
shutil.copy(cloud_observation_filepath, config_dir)
cloud_observation_filepath = config_dir / cloud_observation_filepath.name

# copy the config file to the raw directory and use it
shutil.copy(configfile, config_dir)
configfile = config_dir / configfile.name


sharepath_individual = rawdirectory_individual / "share"
sharepath_individual.mkdir(exist_ok=True)


# path and file names for output data
updated_configfile["outputdata"].update(
    setup_filename=str(config_dir / "eurec4a1d_setup.txt"),
    stats_filename=str(config_dir / "eurec4a1d_stats.txt"),
    zarrbasedir=str(rawdirectory_individual / "eurec4a1d_sol.zarr"),
)

setupfile = updated_configfile["outputdata"]["setup_filename"]
dataset = updated_configfile["outputdata"]["zarrbasedir"]

updated_configfile["coupled_dynamics"].update(
    **dict(
        type="fromfile",  # type of coupled dynamics to configure
        press=str(
            sharepath_individual / "eurec4a1d_dimlessthermo_press.dat"
        ),  # binary filename for pressure
        temp=str(
            sharepath_individual / "eurec4a1d_dimlessthermo_temp.dat"
        ),  # binary filename for temperature
        qvap=str(
            sharepath_individual / "eurec4a1d_dimlessthermo_qvap.dat"
        ),  # binary filename for vapour mixing ratio
        qcond=str(
            sharepath_individual / "eurec4a1d_dimlessthermo_qcond.dat"
        ),  # binary filename for liquid mixing ratio
        wvel=str(
            sharepath_individual / "eurec4a1d_dimlessthermo_wvel.dat"
        ),  # binary filename for vertical (coord3) velocity
        thermo=str(
            sharepath_individual / "eurec4a1d_dimlessthermo.dat"
        ),  # binary filename for thermodynamic profiles
    )
)

updated_configfile["inputfiles"].update(
    grid_filename=str(
        sharepath_individual / "eurec4a1d_ddimlessGBxboundaries.dat"
    ),  # binary filename for initialisation of GBxs / GbxMaps
    constants_filename=str(
        path2CLEO / "libs/cleoconstants.hpp"
    ),  # filename for constants
)
updated_configfile["initsupers"].update(
    initsupers_filename=str(sharepath_individual / "eurec4a1d_dimlessSDsinit.dat")
)

gridfile = updated_configfile["inputfiles"]["grid_filename"]
initSDsfile = updated_configfile["initsupers"]["initsupers_filename"]
thermofile = updated_configfile["coupled_dynamics"]["thermo"]


# %%
### ---------------------------------------------------------------- ###
### --- SETTING UP THERMODYNAMICS AND SUPERDROPLET INITIAL SETUP --- ###
### ---------------------------------------------------------------- ###


### --- settings for 1-D gridbox boundaries --- ###
cloud_altitude = cloud_observation_config["cloud"]["altitude"][0]
# only use integer precision
cloud_altitude = int(cloud_altitude)

cloud_bottom = cloud_altitude - 100
cloud_top = cloud_altitude + 100
vertical_resolution = 20

# zgrid       = [0, cloud_top, vertical_resolution]      # evenly spaced zhalf coords [zmin, zmax, zdelta] [m]
zgrid = np.arange(0, cloud_bottom, vertical_resolution)
zgrid = np.append(zgrid, np.mean([cloud_bottom, cloud_top]))

xgrid = np.array([0, 20])  # array of xhalf coords [m]
ygrid = np.array([0, 20])  # array of yhalf coords [m]

air_temperature_params = cloud_observation_config["thermodynamics"]["air_temperature"][
    "parameters"
]
specific_humidity_params = cloud_observation_config["thermodynamics"][
    "specific_humidity"
]["parameters"]

### --- settings for 1-D Thermodynamics --- ###
PRESS0 = 101315  # [Pa]
TEMP0 = air_temperature_params["f_0"][0]  # [K]
TEMPlapses = (
    np.array(air_temperature_params["slopes"]) * -1e3
)  # -1e3 due to conversion from dT/dz [K/m] to -dT/dz [K/km]
qvap0 = specific_humidity_params["f_0"][0]  # [Kg/Kg]
qvaplapses = (
    np.array(specific_humidity_params["slopes"]) * -1e6
)  # -1e6 due to conversion from dvap/dz [kg/kg m^-1] to -dvap/dz [g/Kg km^-1]
qcond = 0.0  # [Kg/Kg]
WVEL = 0.0  # [m/s]
Wlength = (
    0  # [m] use constant W (Wlength=0.0), or sinusoidal 1-D profile below cloud base
)

z_split_temp = air_temperature_params["x_split"]  # [m]
z_split_qvap = specific_humidity_params["x_split"]  # [m]

# create the base of the cloud as the mean of the two splits
COORD3LIM = int(cloud_bottom - vertical_resolution)  # [m]

### --- settings for initial superdroplets --- ###
npergbx = updated_configfile["initsupers"][
    "initnsupers"
]  # number of superdroplets per gridbox

# initial superdroplet radii (and implicitly solute masses)
MINRADIUS = 1e-7
MAXRADIUS = 1e-3
rspan = [MINRADIUS, MAXRADIUS]  # min and max range of radii to sample [m]

# initial superdroplet attributes
psd_params = cloud_observation_config["particle_size_distribution"]["parameters"]

# settings for initial superdroplet multiplicies with ATR and Aerosol from Lohmann et. al 2016 Fig. 5.5
geomeans = psd_params["geometric_means"]
geosigs = psd_params["geometric_sigmas"]
scalefacs = psd_params["scale_factors"]
numconc = np.sum(scalefacs)

### ---------------------------------------------------------------- ###
### UPDATE THE BOUNDARY CONDITIONS FOR THE CONFIG FILE ###
### ---------------------------------------------------------------- ###

updated_configfile["boundary_conditions"].update(
    COORD3LIM=COORD3LIM,  # SDs added to domain with coord3 >= COORD3LIM [m]
    newnsupers=npergbx,  # number of new super-droplets per gridbox
    MINRADIUS=MINRADIUS,  # minimum radius of new super-droplets [m]
    MAXRADIUS=MAXRADIUS,  # maximum radius of new super-droplets [m]
    NUMCONC_a=scalefacs[0],  # number conc. of 1st droplet lognormal dist [m^-3]
    GEOMEAN_a=geomeans[0],  # geometric mean radius of 1st lognormal dist [m]
    geosigma_a=geosigs[0],  # geometric standard deviation of 1st lognormal dist
    NUMCONC_b=scalefacs[1],  # number conc. of 2nd droplet lognormal dist [m^-3]
    GEOMEAN_b=geomeans[1],  # geometric mean radius of 2nd lognormal dist [m]
    geosigma_b=geosigs[1],  # geometric standard deviation of 2nd lognormal dist
)


### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
# %%
### ---------------------------------------------------------------- ###
### ----- MODIFY THE CONFIG FILE PRIOR TO CREATING INPUT FILES ----- ###
### ---------------------------------------------------------------- ###

editconfigfile.edit_config_params(str(configfile), updated_configfile)

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
    Path(sharepath).mkdir(exist_ok=True)
    Path(binpath).mkdir(exist_ok=True)
os.system("rm " + gridfile)
os.system("rm " + initSDsfile)
os.system("rm " + thermofile[:-4] + "*")


### ----- write gridbox boundaries binary ----- ###
cgrid.write_gridboxboundaries_binary(gridfile, zgrid, xgrid, ygrid, constsfile)
with Capturing() as grid_info:
    rgrid.print_domain_info(constsfile, gridfile)

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
    updated_configfile["timesteps"]["T_END"]
    / updated_configfile["timesteps"]["MOTIONTSTEP"]
)
# add 1000 to ensure enough space for new SDs
max_number_supers = int(math.ceil(renew_timesteps * npergbx + 1000))
# get the total number of gridboxes
updated_configfile["domain"].update(
    nspacedims=1, ngbxs=total_number_gridboxes, maxnsupers=max_number_supers
)

# update the config file
editconfigfile.edit_config_params(str(configfile), updated_configfile)

# %%

### ----- write thermodynamics binaries ----- ###
thermodyngen = thermogen.ConstHydrostaticLapseRates(
    configfile=configfile,
    constsfile=constsfile,
    PRESS0=PRESS0,
    TEMP0=TEMP0,
    qvap0=qvap0,
    Zbase=COORD3LIM,
    TEMPlapses=TEMPlapses,
    qvaplapses=qvaplapses,
    qcond=qcond,
    WMAX=WVEL,
    UVEL=None,
    VVEL=None,
    Wlength=Wlength,
)
cthermo.write_thermodynamics_binary(
    thermofile, thermodyngen, configfile, constsfile, gridfile
)

### ----- write initial superdroplets binary ----- ###
nsupers = crdgens.nsupers_at_domain_top(gridfile, constsfile, npergbx, COORD3LIM)

# get total number of superdroplets
total_nsupers = int(np.sum(list(nsupers.values())))
updated_configfile["initsupers"].update(initnsupers=total_nsupers)
# update the config file
editconfigfile.edit_config_params(str(configfile), updated_configfile)


# create initial superdroplets coordinates
coord3gen = crdgens.SampleCoordGen(True)  # sample coord3 randomly
coord1gen = None  # do not generate superdroplet coord2s
coord2gen = None  # do not generate superdroplet coord2s

# create initial superdroplets attributes
xiprobdist = probdists.LnNormal(geomeans, geosigs, scalefacs)
radiigen = rgens.SampleLog10RadiiGen(rspan)

# create uniform dry radii
monodryr = 1e-9  # all SDs have this same dryradius [m]
dryradiigen = rgens.MonoAttrGen(monodryr)

# write initial superdroplets binary
initattrsgen = attrsgen.AttrsGenerator(
    radiigen, dryradiigen, xiprobdist, coord3gen, coord1gen, coord2gen
)
csupers.write_initsuperdrops_binary(
    initSDsfile, initattrsgen, configfile, constsfile, gridfile, nsupers, numconc
)
# ### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###

### ---------------------------------------------------------------- ###
### --------- MODIFY THE CONFIG FILE PRIOR TO RUNNING CLEO --------- ###
### ---------------------------------------------------------------- ###

editconfigfile.edit_config_params(str(configfile), updated_configfile)

### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
### ---------------------- PLOT INIT FIGURES ----------------------- ###
### ---------------------------------------------------------------- ###

isfigures = [True, True]  # booleans for [making, saving] initialisation figures
savefigpath = (
    str(rawdirectory_individual / "figures") + "/"
)  # directory for saving figures

SDgbxs2plt = total_number_gridboxes - 1

### ----- show (and save) plots of binary file data ----- ###
if isfigures[0]:
    if isfigures[1]:
        Path(savefigpath).mkdir(exist_ok=True)
    rgrid.plot_gridboxboundaries(constsfile, gridfile, savefigpath, isfigures[1])
    rthermo.plot_thermodynamics(
        constsfile, configfile, gridfile, thermofile, savefigpath, isfigures[1]
    )
    rsupers.plot_initGBxs_distribs(
        configfile,
        constsfile,
        initSDsfile,
        gridfile,
        savefigpath,
        isfigures[1],
        SDgbxs2plt,
    )
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###


### ---------------------------------------------------------------- ###
### ---------------------- RUN CLEO EXECUTABLE --------------------- ###
### ---------------------------------------------------------------- ###
print("===============================")
print("Running CLEO executable")
os.chdir(path2build)
os.system("pwd")
os.system("rm -rf " + dataset)  # delete any existing dataset
executable = str(path2build) + "/examples/eurec4a1d/src/eurec4a1D"
print("Executable: " + executable)
print("Config file: " + str(configfile))
os.system(executable + " " + str(configfile))
print("===============================")
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
