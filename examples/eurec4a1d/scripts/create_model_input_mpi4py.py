# %%
import random
import re
import sys
import shutil
import yaml
import math
from typing import Union, Tuple
from io import StringIO
from pathlib import Path
import logging
#%%
import sys
import numpy as np
import mpi4py
import time as pytime
import pandas as pd
import xarray as xr
import logging
#%%
# logging configure
logging.basicConfig(level=logging.INFO)

# === mpi4py ===
try:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()  # [0,1,2,3,4,5,6,7,8,9]
  npro = comm.Get_size()  # 10
except:
  print('::: Warning: Proceeding without mpi4py! :::')
  rank = 0
  npro = 1

path2CLEO = Path(__file__).resolve().parents[3]

path2build = path2CLEO / "build_eurec4a1d"
path2eurec4a1d = path2CLEO / "examples/eurec4a1d"

import datetime
# === logging ===
# create log file

time_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')

log_file_dir = path2eurec4a1d / "logfiles" / f"create_init_files/mpi4py/{time_str}"
log_file_dir.mkdir(exist_ok=True, parents=True)
log_file_path = log_file_dir / f"{rank}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(log_file_path)
handler.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(console_handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Execution terminated due to an Exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


logging.info(f"====================")
logging.info(f"Start with rank {rank} of {npro}")

binary_dir_path = path2build / "bin/"
share_dir_path = path2build / "share/"

constants_file_path = path2CLEO / "libs/cleoconstants.hpp"
origin_config_file_path = path2eurec4a1d / "default_config/eurec4a1d_config_stationary.yaml"
breakup_config_file_path = path2eurec4a1d / "default_config/breakup.yaml"

output_dir_path = path2CLEO / "data/debug_output"
output_dir_path.mkdir(exist_ok=True, parents=True)

from sdm_eurec4a import RepositoryPath
path2sdm_eurec4a = RepositoryPath('levante').data_dir
input_dir_path = path2sdm_eurec4a / "model/input_v4.0"



# %%

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr

import sdm_eurec4a.input_processing.models as smodels
from sdm_eurec4a.conversions import msd_from_psd_dataarray


if "sdm_pysd_env312" not in sys.prefix:
    sys.path.append(str(path2CLEO))  # for imports from pySD package
    sys.path.append(
        str(path2CLEO / "examples/exampleplotting/")
    )  # for imports from example plotting package


from pySD import editconfigfile
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


# print(f"Enviroment: {sys.prefix}")


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


def parameters_dataset_to_dict(ds: xr.Dataset, mapping : Union[dict[str,str], Tuple[str]]) -> dict:
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
    else :
        raise TypeError('mapping must be a dict or a tuple')

    return parameters


def plot_11(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    g_min = min(x_min, y_min)
    g_max = max(x_max, y_max)
    g = (g_max, g_min)

    ax.plot(g, g, color="black", linestyle="--")


if "sdm_pysd_env312" not in sys.prefix:
    sys.path.append(str(path2CLEO))  # for imports from pySD package
    sys.path.append(
        str(path2CLEO / "examples/exampleplotting/")
    )  # for imports from example plotting package



# if False :
#     path2CLEO = Path(sys.argv[1])
#     origin_config_file_path = Path(sys.argv[2])
#     origin_cloud_observation_file = Path(sys.argv[3])

#     raw_dir = Path(sys.argv[4])

#     breakup_config_file = sys.argv[5]
#     if breakup_config_file in ("None", 0, "0", None, False):
#         breakup_config_file = None
#     elif not Path(breakup_config_file).exists():
#         raise ValueError(f"Breakup config file not found under: {breakup_config_file}")
#     else:
#         breakup_config_file = Path(breakup_config_file)
# else :



# print(f"Path to CLEO: {path2CLEO}")
# print(f"Path to config file: {origin_config_file_path}")
# print(f"Path to parameter input files: {input_dir_path}")
# print(f"Path to output directory: {output_dir_path}")
# print(f"Path to breakup config file: {breakup_config_file_path}")

# %%
# Load the parameters datasets
ds_psd_parameters = xr.open_dataset(input_dir_path / "particle_size_distribution_parameters_linear_space.nc")
ds_potential_temperature_parameters = xr.open_dataset(input_dir_path / "potential_temperature_parameters.nc")
ds_relative_humidity_parameters = xr.open_dataset(input_dir_path / "relative_humidity_parameters.nc")
ds_pressure_parameters = xr.open_dataset(input_dir_path / "pressure_parameters.nc")

xr.testing.assert_allclose(ds_potential_temperature_parameters['x_split'], ds_relative_humidity_parameters['x_split'], rtol = 1e-12)

# convert parameters from log space to linear space


mapping = dict(
    geometric_mean1 = 'geometric_mean1',
    geometric_mean2 = 'geometric_mean2',
    geometric_std_dev1 = 'geometric_std_dev1',
    geometric_std_dev2 = 'geometric_std_dev2',
    scale_factor1 = 'scale_factor1',
    scale_factor2 = 'scale_factor2',
    )



# %%

shared_cloud_ids = set.intersection(
    set(ds_psd_parameters['cloud_id'].values),
    set(ds_potential_temperature_parameters['cloud_id'].values),
    set(ds_relative_humidity_parameters['cloud_id'].values),
    set(ds_pressure_parameters['cloud_id'].values),
)
shared_cloud_ids = list(sorted(shared_cloud_ids))
# %%

# create the individual raw directory
identification_type = "cluster"


# GENERAL SETTINGS FOR ALL CLOUDS


sublist_cloud_ids = np.array_split(shared_cloud_ids, npro)[rank]

for step, cloud_id in enumerate(sublist_cloud_ids):

    logging.info(f"Core {rank+1} {step}/{len(sublist_cloud_ids)} Cloud {cloud_id}")

    # --- extract cloud specific parameters --- #
    psd_params = ds_psd_parameters.sel(cloud_id = cloud_id)
    psd_params_dict = parameters_dataset_to_dict(psd_params, mapping)

    relative_humidity_params = ds_relative_humidity_parameters.sel(cloud_id = cloud_id)
    potential_temperature_params = ds_potential_temperature_parameters.sel(cloud_id = cloud_id)
    pressure_params = ds_pressure_parameters.sel(cloud_id = cloud_id)

    logging.info(f"Read default config file from {origin_config_file_path}")
    # CREATE A CONFIG FILE TO BE UPDATED
    with open(origin_config_file_path, "r") as f:
        eurec4a1d_config = yaml.safe_load(f)

    # update breakup in eurec4a1d_config file if breakup file is given:
    logging.info(f"Read breakup config file from {breakup_config_file_path}")
    if breakup_config_file_path is not None:
        with open(breakup_config_file_path, "r") as f:
            breakup_config = yaml.safe_load(f)
        eurec4a1d_config["microphysics"].update(breakup_config)

    individual_output_dir_path = output_dir_path / f"{identification_type}_{cloud_id}"
    individual_output_dir_path.mkdir(exist_ok=True, parents= False)


    logging.info(f"Copy config file to {individual_output_dir_path}")
    config_dir_path = individual_output_dir_path / "config"
    config_dir_path.mkdir(exist_ok=True, parents= False)
    # copy the cloud config file to the raw directory and use it
    config_file_path = config_dir_path / "eurec4a1d_config.yaml"
    shutil.copy(origin_config_file_path, config_file_path)

    logging.info(f"Create share directory {individual_output_dir_path}")
    share_path_individual = individual_output_dir_path / "share"
    share_path_individual.mkdir(exist_ok=True)


    # --- INPUT DATA ---
    logging.info(f"Update input data in config file")
    # coupling dynamics files
    eurec4a1d_config["coupled_dynamics"].update(
        **dict(
            type="fromfile",  # type of coupled dynamics to configure
            # binary filename for pressure
            press=str(share_path_individual / "eurec4a1d_dimlessthermo_press.dat"),
            # binary filename for temperature
            temp=str(share_path_individual / "eurec4a1d_dimlessthermo_temp.dat"),
            # binary filename for vapour mixing ratio
            qvap=str(share_path_individual / "eurec4a1d_dimlessthermo_qvap.dat"),
            # binary filename for liquid mixing ratio
            qcond=str(share_path_individual / "eurec4a1d_dimlessthermo_qcond.dat"),
            # binary filename for vertical (coord3) velocity
            wvel=str(share_path_individual / "eurec4a1d_dimlessthermo_wvel.dat"),
            # binary filename for thermodynamic profiles
            thermo=str(share_path_individual / "eurec4a1d_dimlessthermo.dat"),
        )
    )

    # input files of gridbox boundaries and initial superdroplets
    eurec4a1d_config["inputfiles"].update(
        # binary filename for initialisation of GBxs / GbxMaps
        grid_filename=str(share_path_individual / "eurec4a1d_ddimlessGBxboundaries.dat"),
        # filename for constants
        constants_filename=str(path2CLEO / "libs/cleoconstants.hpp"),
    )
    eurec4a1d_config["initsupers"].update(
        # binary filename for initial superdroplets
        initsupers_filename=str(share_path_individual / "eurec4a1d_dimlessSDsinit.dat")
    )

    grid_file_path = eurec4a1d_config["inputfiles"]["grid_filename"]
    init_superdroplets_file_path = eurec4a1d_config["initsupers"]["initsupers_filename"]
    thermodynamics_file_path = eurec4a1d_config["coupled_dynamics"]["thermo"]

    # --- OUTPUT DATA ---
    logging.info(f"Update output data in config file")
    eurec4a1d_config["outputdata"].update(
        setup_filename=str(config_dir_path / "eurec4a1d_setup.txt"),
        stats_filename=str(config_dir_path / "eurec4a1d_stats.txt"),
        zarrbasedir=str(individual_output_dir_path / "eurec4a1d_sol.zarr"),
    )

    editconfigfile.edit_config_params(str(config_file_path), eurec4a1d_config)

    setup_file_path = eurec4a1d_config["outputdata"]["setup_filename"]
    stats_file_path = eurec4a1d_config["outputdata"]["stats_filename"]
    dataset_file_path = eurec4a1d_config["outputdata"]["zarrbasedir"]


    ### --- settings for 1-D gridbox boundaries --- ###
    # only use integer precision
    cloud_altitude = potential_temperature_params['x_split'].mean().values
    cloud_altitude = int(cloud_altitude)

    dz = 20
    dz_cloud = 100
    dx = 100
    dy = 100

    cloud_bottom = cloud_altitude - dz_cloud / 2

    # below cloud
    zgrid = np.arange(0, cloud_bottom + dz, dz)

    # above cloud
    zgrid_cloud_base = np.max(zgrid)
    zgrid_cloud_top = zgrid_cloud_base + dz_cloud
    zgrid = np.append(zgrid, zgrid_cloud_top)

    xgrid = np.array([0, dx])  # array of xhalf coords [m]
    ygrid = np.array([0, dy])  # array of yhalf coords [m]

    # create initial superdroplets coordinates
    coord3gen = crdgens.SampleCoordGen(True)  # sample coord3 randomly
    coord1gen = None  # do not generate superdroplet coord2s
    coord2gen = None  # do not generate superdroplet coord2s


    ### --- settings for initial superdroplets --- ###
    # number of superdroplets per gridbox
    sd_per_gridbox = eurec4a1d_config["initsupers"]["initnsupers"]

    # initial superdroplet radii (and implicitly solute masses)
    radius_minimum = 50e-6
    radius_maximum = 3e-3
    radius_span = [
        radius_minimum,
        radius_maximum,
    ]  # min and max range of radii to sample [m]


    # create initial superdroplets attributes
    radii_generator = rgens.SampleLog10RadiiWithBinWidth(radius_span)
    # create uniform dry radii
    monodryr = 1e-12  # all SDs have this same dryradius [m]
    dryradii_generator = rgens.MonoAttrGen(monodryr)

    logging.info("Write gridbox binary file")
    ### ----- write gridbox boundaries binary ----- ###
    with Capturing() as grid_info:
        cgrid.write_gridboxboundaries_binary(
            gridfile= grid_file_path,
            zgrid= zgrid,
            xgrid= xgrid,
            ygrid= ygrid,
            constsfile= constants_file_path,
            )
    with Capturing() as grid_info:
        rgrid.print_domain_info(constants_file_path, grid_file_path)

    # extract the total number of gridboxes
    for line in grid_info:
        if "domain no. gridboxes:" in line:
            grid_dimensions = np.array(
                line.split(":")[-1].replace(" ", "").split("x"), dtype=int
            )
            number_gridboxes_total = int(np.prod(grid_dimensions))
            break

    # --- THERMODYNAMICS ---

    logging.info("Create thermodynamics generator")
    thermodynamics_generator = thermogen.SplittedLapseRates(
        configfile = config_file_path,
        constsfile = constants_file_path,
        cloud_base_height = relative_humidity_params['x_split'].values,
        pressure_0 = pressure_params['f_0'].values,
        potential_temperature_0 = potential_temperature_params['f_0'].values,
        relative_humidity_0 = relative_humidity_params['f_0'].values,
        pressure_lapse_rates = (
            pressure_params['slope'].values,
            pressure_params['slope'].values,
        ),
        potential_temperature_lapse_rates = (
            potential_temperature_params['slope_1'].values,
            potential_temperature_params['slope_2'].values,
        ),
        relative_humidity_lapse_rates = (
            relative_humidity_params['slope_1'].values,
            relative_humidity_params['slope_2'].values,
        ),
        qcond=0.0,
        w_maximum = 0.0,
        u_velocity = None,
        v_velocity = None,
        Wlength = 0.0,
    )

    logging.info("Write thermodynamics binary")
    with Capturing() as thermo_info:
        cthermo.write_thermodynamics_binary(
            thermofile= thermodynamics_file_path,
            thermogen= thermodynamics_generator,
            configfile= config_file_path,
            constsfile= constants_file_path,
            gridfile= grid_file_path,
        )


    # --- INITIAL SUPERDROPLETS ---
    logging.info("Create initial multiplicity generator")
    xi_probability_distribution = probdists.DoubleLogNormal(
        geometric_mean1= psd_params_dict['geometric_mean1'],
        geometric_mean2= psd_params_dict['geometric_mean2'],
        geometric_std_dev1= psd_params_dict['geometric_std_dev1'],
        geometric_std_dev2= psd_params_dict['geometric_std_dev2'],
        scale_factor1= psd_params_dict['scale_factor1'],
        scale_factor2= psd_params_dict['scale_factor2'],
    )

    logging.info("Create initial attributes generator")
    initial_attributes_generator = attrsgen.AttrsGeneratorBinWidth(
        radiigen = radii_generator,
        dryradiigen = dryradii_generator,
        xiprobdist= xi_probability_distribution,
        coord3gen= coord3gen,
        coord1gen= coord1gen,
        coord2gen= coord2gen,
    )

    logging.info("Get superdroplets at domain top")
    ### ----- write initial superdroplets binary ----- ###
    with Capturing() as super_top_info:
        number_superdroplets = crdgens.nsupers_at_domain_top(
            gridfile= grid_file_path,
            constsfile= constants_file_path,
            nsupers = sd_per_gridbox,
            zlim = zgrid_cloud_base,
        )

    # get total number of superdroplets
    number_superdroplets_total = int(np.sum(list(number_superdroplets.values())))
    eurec4a1d_config["initsupers"].update(initnsupers=number_superdroplets_total)

    # Update the max number of superdroplets
    renew_timesteps = (
        eurec4a1d_config["timesteps"]["T_END"]
        / eurec4a1d_config["timesteps"]["MOTIONTSTEP"]
    )
    # add 1000 to ensure enough space for new SDs
    max_number_supers = int(math.ceil(renew_timesteps * sd_per_gridbox + 1000))
    # get the total number of gridboxes
    eurec4a1d_config["domain"].update(
        nspacedims=1,
        ngbxs=number_gridboxes_total,
        maxnsupers=max_number_supers
    )

    editconfigfile.edit_config_params(str(config_file_path), eurec4a1d_config)

    # --- WRITE THE BINARY FILES ---
    logging.info("Write initial superdroplets binary")
    with Capturing() as super_info:
        try :
            csupers.write_initsuperdrops_binary(
                initsupersfile = init_superdroplets_file_path,
                initattrsgen = initial_attributes_generator,
                configfile = config_file_path,
                constsfile = constants_file_path,
                gridfile = grid_file_path,
                nsupers = number_superdroplets,
                NUMCONC = 0,
            )
        except Exception as e:
            logging.error(f"Error: {type(e)}")


    ### ---------------------------------------------------------------- ###
    ### UPDATE THE BOUNDARY CONDITIONS FOR THE CONFIG FILE ###
    ### ---------------------------------------------------------------- ###
    logging.info("Update boundary conditions in config file")
    eurec4a1d_config["boundary_conditions"].update(
        COORD3LIM=float(zgrid_cloud_base),  # SDs added to domain with coord3 >= z_boundary_respawn [m]
        newnsupers=sd_per_gridbox,  # number of new super-droplets per gridbox
        MINRADIUS=radius_minimum,  # minimum radius of new super-droplets [m]
        MAXRADIUS=radius_maximum,  # maximum radius of new super-droplets [m]
        NUMCONC_a=psd_params_dict['scale_factor1'],  # number conc. of 1st droplet lognormal dist [m^-3]
        GEOMEAN_a=psd_params_dict['geometric_mean1'],  # geometric mean radius of 1st lognormal dist [m]
        geosigma_a=psd_params_dict['geometric_std_dev1'],  # geometric standard deviation of 1st lognormal dist
        NUMCONC_b=psd_params_dict['scale_factor2'],  # number conc. of 2nd droplet lognormal dist [m^-3]
        GEOMEAN_b=psd_params_dict['geometric_mean2'],  # geometric mean radius of 2nd lognormal dist [m]
        geosigma_b=psd_params_dict['geometric_std_dev2'],  # geometric standard deviation of 2nd lognormal dist
    )


    # --- PLOTTING ---

    logging.info("Plot figures")
    fig_dir = individual_output_dir_path / "figures"
    fig_dir.mkdir(exist_ok=True, parents=False)

    gridbox_to_plot = number_gridboxes_total - 1

    isfigures = [True, True]  # booleans for [making, saving] initialisation figures
    ### ----- show (and save) plots of binary file data ----- ###
    with Capturing() as plot_info:
        if isfigures[0]:
            try:
                rgrid.plot_gridboxboundaries(
                    constsfile= constants_file_path,
                    gridfile= grid_file_path,
                    binpath= str(fig_dir),
                    savefig= isfigures[1])
            except Exception as e:
                logging.error(f"Error: {type(e)}")
            try:
                rthermo.plot_thermodynamics(
                    constsfile= constants_file_path,
                    configfile= config_file_path,
                    gridfile= grid_file_path,
                    thermofile= thermodynamics_file_path,
                    binpath= str(fig_dir),
                    savefig= isfigures[1],
                )
            except Exception as e:
                logging.error(f"Error: {type(e)}")
            try:
                rsupers.plot_initGBxs_distribs(
                    configfile = config_file_path,
                    constsfile = constants_file_path,
                    initsupersfile = init_superdroplets_file_path,
                    gridfile = grid_file_path,
                    binpath = str(fig_dir),
                    savefig = isfigures[1],
                    gbxs2plt = gridbox_to_plot,
                )
            except Exception as e:
                logging.error(f"Error: {type(e)}")

            plt.close("all")
