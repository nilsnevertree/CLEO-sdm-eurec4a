# %%
import sys
import ruamel.yaml
import secrets
from mpi4py import MPI


yaml = ruamel.yaml.YAML()
import math
from typing import Union, Tuple
from io import StringIO
from pathlib import Path
import logging
import datetime
import numpy as np
import xarray as xr
import argparse
import matplotlib.pyplot as plt

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
from pySD.thermobinary_src import windsgen
from pySD.thermobinary_src import thermodyngen

# %%
# validate the mpi4py setup

# === mpi4py ===
path2CLEO = Path(__file__).resolve().parents[3]
# path2build = path2CLEO / "build_eurec4a1d"
path2eurec4a1d = path2CLEO / "examples/eurec4a1d"

# create shared logging directory
try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # [0,1,2,3,4,5,6,7,8,9]
    number_ranks = comm.Get_size()  # 10
except Exception:
    raise ImportError("MPI4PY not possible!")

# create shared logging directory
if rank == 0:
    # Generate a shared directory name based on UTC time and random hex
    time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    random_hex = secrets.token_hex(4)
    log_dir = (
        path2eurec4a1d
        / "logfiles"
        / f"create_init_files/mpi4py/{time_str}-{random_hex}"
    )
    log_dir.mkdir(exist_ok=True, parents=True)
else:
    log_dir = None


# Broadcast the shared directory name to all processes
log_dir = comm.bcast(log_dir, root=0)
# create individual log file
log_file_path = log_dir / f"{rank}.log"

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
        "Execution terminated due to an Exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


logging.info("====================")
logging.info(f"Start with rank {rank} of {number_ranks}")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--output_dir_path",
    type=str,
    help="Path to output directory which should inhibite the subdirectories for each cloud.",
)
parser.add_argument(
    "--input_dir_path",
    type=str,
    help="Path to directory which contains the input files from which the initial data and configuraion will be created.",
)
parser.add_argument(
    "--breakup_config_file_path",
    type=str,
    help="Path to breakup configuration file.",
)
parser.add_argument(
    "--default_config_file_path",
    type=str,
    help="Number of ranks used for parallel processing.",
)
args = parser.parse_args()


# building paths to default configuration files and the CLEO constants file
constants_file_path = path2CLEO / "libs/cleoconstants.hpp"
logging.info(f"Constants file path: {constants_file_path}")

input_dir_path = Path(args.input_dir_path)
logging.info(f"Input directory: {input_dir_path}")
if not input_dir_path.exists():
    raise ValueError(f"Input directory not found under: {input_dir_path}")

origin_config_file_path = Path(args.default_config_file_path)
logging.info(f"Original config file path: {origin_config_file_path}")

breakup_config_file_path = Path(args.breakup_config_file_path)
logging.info(f"Breakup config file path: {breakup_config_file_path}")


# getting the input and output directories
output_dir_path = Path(args.output_dir_path)
logging.info(f"Output directory: {output_dir_path}")
output_dir_path.mkdir(exist_ok=True, parents=True)

# define the prefix for the individual subfolders created for each cloud.
# the folder name will be the {subfolder_prefix}{cloud_id}
subfolder_prefix = "cluster_"

# # NOTE: test setup for local testing

# output_dir_path = path2CLEO / "data/debug_output"
# output_dir_path.mkdir(exist_ok=True, parents=True)

# from sdm_eurec4a import RepositoryPath
# path2sdm_eurec4a = RepositoryPath('levante').data_dir
# input_dir_path = path2sdm_eurec4a / "model/input_v4.0"


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
    Convert selected parameters from an xarray Dataset to a dictionary with a given mapping.

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


# Load the parameters datasets from the input directory
logging.info("Load parameters datasets")
# linear space for the particle size distribution parameters with data variables
# - geometric_mean1, geometric_std_dev1, scale_factor1
# - geometric_mean2, geometric_std_dev2, scale_factor2
ds_psd_parameters = xr.open_dataset(
    input_dir_path / "particle_size_distribution_parameters_linear_space.nc"
)
# potential temperature, relative humidity with parameters
# - f_0, slope_1, slope_2
ds_potential_temperature_parameters = xr.open_dataset(
    input_dir_path / "potential_temperature_parameters.nc"
)
ds_relative_humidity_parameters = xr.open_dataset(
    input_dir_path / "relative_humidity_parameters.nc"
)
ds_pressure_parameters = xr.open_dataset(input_dir_path / "pressure_parameters.nc")

logging.info("Testing datasets have equal x_split values")
xr.testing.assert_allclose(
    ds_potential_temperature_parameters["x_split"],
    ds_relative_humidity_parameters["x_split"],
    rtol=1e-12,
)


# mapping of the parameters to extract from the datasets of the input directory
# this mapping assumes a double log normal distribution for the particle size distribution
mapping = dict(
    geometric_mean1="geometric_mean1",
    geometric_mean2="geometric_mean2",
    geometric_std_dev1="geometric_std_dev1",
    geometric_std_dev2="geometric_std_dev2",
    scale_factor1="scale_factor1",
    scale_factor2="scale_factor2",
)

logging.info("Find shared cloud ids from the input datasets")
shared_cloud_ids = set.intersection(
    set(ds_psd_parameters["cloud_id"].values),
    set(ds_potential_temperature_parameters["cloud_id"].values),
    set(ds_relative_humidity_parameters["cloud_id"].values),
    set(ds_pressure_parameters["cloud_id"].values),
)
shared_cloud_ids = list(sorted(shared_cloud_ids))

# split the cloud ids among the processes and select the sublist for the current rank
sublist_cloud_ids = np.array_split(shared_cloud_ids, number_ranks)[rank]

# iterate over the cloud ids
logging.info("Start creating the initial data for the process specific clouds")
for step, cloud_id in enumerate(sublist_cloud_ids):
    logging.info(f"Rank {rank+1} {step}/{len(sublist_cloud_ids)} Cloud {cloud_id}")

    # --- extract cloud specific parameters --- #
    psd_params = ds_psd_parameters.sel(cloud_id=cloud_id)
    psd_params_dict = parameters_dataset_to_dict(psd_params, mapping)

    relative_humidity_params = ds_relative_humidity_parameters.sel(cloud_id=cloud_id)
    potential_temperature_params = ds_potential_temperature_parameters.sel(
        cloud_id=cloud_id
    )
    pressure_params = ds_pressure_parameters.sel(cloud_id=cloud_id)

    logging.info(f"Read default config file from {origin_config_file_path}")
    # CREATE A CONFIG FILE TO BE UPDATED
    with open(origin_config_file_path, "r") as f:
        eurec4a1d_config = yaml.load(f)

    # update breakup in eurec4a1d_config file if breakup file is given:
    logging.info(f"Read breakup config file from {breakup_config_file_path}")
    if breakup_config_file_path is not None:
        with open(breakup_config_file_path, "r") as f:
            breakup_config = yaml.load(f)
        eurec4a1d_config["microphysics"].update(breakup_config)

    individual_output_dir_path = output_dir_path / f"{subfolder_prefix}{cloud_id}"
    individual_output_dir_path.mkdir(exist_ok=True, parents=False)

    # copy config files to the individual output directory
    config_dir_path = individual_output_dir_path / "config"
    config_dir_path.mkdir(exist_ok=True, parents=False)
    # copy the cloud config file to the raw directory and use it
    config_file_path = config_dir_path / "eurec4a1d_config.yaml"
    logging.info(f"Copy config file to {config_file_path}")
    with open(config_file_path, "w") as f:
        yaml.dump(eurec4a1d_config, f)

    logging.info(f"Create share directory {individual_output_dir_path}")
    share_path_individual = individual_output_dir_path / "share"
    share_path_individual.mkdir(exist_ok=True)

    # --- INPUT DATA ---
    logging.info("Update input data in config file")

    grid_file_path = share_path_individual / "eurec4a1d_ddimlessGBxboundaries.dat"
    init_superdroplets_file_path = (
        share_path_individual / "eurec4a1d_dimlessSDsinit.dat"
    )
    thermodynamics_file_path = share_path_individual / "eurec4a1d_dimlessthermo.dat"

    setup_file_path = config_dir_path / "eurec4a1d_setup.txt"
    stats_file_path = config_dir_path / "eurec4a1d_stats.txt"
    dataset_file_path = individual_output_dir_path / "eurec4a1d_sol.zarr"

    thermofile_dict = dict(
        [
            (
                var,
                (
                    share_path_individual / f"eurec4a1d_dimlessthermo_{var}.dat"
                ).as_posix(),
            )
            for var in ["press", "temp", "qvap", "qcond", "wvel"]
        ]
    )

    # coupling dynamics files
    eurec4a1d_config["coupled_dynamics"].update(
        thermo=thermodynamics_file_path.as_posix(),  # binary filename for thermodynamic profiles
        **thermofile_dict,
    )

    # input files of gridbox boundaries and initial superdroplets
    eurec4a1d_config["inputfiles"].update(
        grid_filename=grid_file_path.as_posix(),  # binary filename for initialisation of GBxs / GbxMaps
        constants_filename=constants_file_path.as_posix(),  # filename for constants
    )
    eurec4a1d_config["initsupers"].update(
        initsupers_filename=init_superdroplets_file_path.as_posix()  # binary filename for initial superdroplets
    )

    # --- OUTPUT DATA ---
    logging.info("Update output data in config file")
    eurec4a1d_config["outputdata"].update(
        setup_filename=setup_file_path.as_posix(),  # filename for setup file
        stats_filename=stats_file_path.as_posix(),  # filename for stats file
        zarrbasedir=dataset_file_path.as_posix(),  # base directory for zarr output
    )

    editconfigfile.edit_config_params(config_file_path, eurec4a1d_config)

    ### --- settings for 1-D gridbox boundaries --- ###
    # only use integer precision
    cloud_altitude = potential_temperature_params["x_split"].mean().values
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

    # wind field+
    WVEL = 0.0  # [m/s]
    Wlength = zgrid_cloud_base

    ### ----- write gridbox boundaries binary ----- ###
    logging.info("Write gridbox binary file")
    cgrid.write_gridboxboundaries_binary(
        grid_filename=grid_file_path,
        zgrid=zgrid,
        xgrid=xgrid,
        ygrid=ygrid,
        constants_filename=constants_file_path,
    )

    # use the gridbox boundaries binary file to get the number of gridboxes
    logging.info("Get number of gridboxes from gridbox binary file")
    try:
        vols = rgrid.get_gbxvols_from_gridfile(
            grid_filename=grid_file_path,
            constants_filename=constants_file_path,
            isprint=True,
        )
        number_gridboxes_total = np.size(vols)
    except Exception as e:
        logging.info("Number of gridboxes not found due to exception")
        raise e

    ### ----- write thermodynamics binaries ----- ###
    logging.info("Create thermodynamics generator")
    thermo_generator = thermogen.SplittedLapseRates(
        config_filename=config_file_path,
        constants_filename=constants_file_path,
        cloud_base_height=relative_humidity_params["x_split"].values,  # type: ignore
        pressure_0=pressure_params["f_0"].values,  # type: ignore
        pressure_reference=100000,  # use 1000 hPa as reference pressure [Pa] as in AMESOC (https://glossary.ametsoc.org/wiki/Potential_temperature)
        potential_temperature_0=potential_temperature_params["f_0"].values,  # type: ignore
        relative_humidity_0=relative_humidity_params["f_0"].values,  # type: ignore
        pressure_lapse_rates=(  # type: ignore
            pressure_params["slope"].values,  # type: ignore
            pressure_params["slope"].values,  # type: ignore
        ),  # type: ignore
        potential_temperature_lapse_rates=(  # type: ignore
            potential_temperature_params["slope_1"].values,  # type: ignore
            potential_temperature_params["slope_2"].values,  # type: ignore
        ),  # type: ignore
        relative_humidity_lapse_rates=(  # type: ignore
            relative_humidity_params["slope_1"].values,  # type: ignore
            relative_humidity_params["slope_2"].values,  # type: ignore
        ),
        qcond=0.0,
    )
    # create wind generator
    winds_generator = windsgen.SinusoidalUpdraught(WVEL, None, None, Wlength)
    thermodynamic_generator = thermodyngen.ThermodynamicsGenerator(
        thermo_generator, winds_generator
    )
    # thermodynamic generator
    logging.info("Write thermodynamics binary")
    with Capturing() as thermo_info:
        cthermo.write_thermodynamics_binary(
            thermofiles=thermodynamics_file_path,
            thermodyngen=thermodynamic_generator,
            config_filename=config_file_path,
            constants_filename=constants_file_path,
            grid_filename=grid_file_path,
        )

    # --- INITIAL SUPERDROPLETS ---
    logging.info("Create initial multiplicity generator")
    xi_probability_distribution = probdists.DoubleLogNormal(
        geometric_mean1=psd_params_dict["geometric_mean1"],
        geometric_mean2=psd_params_dict["geometric_mean2"],
        geometric_std_dev1=psd_params_dict["geometric_std_dev1"],
        geometric_std_dev2=psd_params_dict["geometric_std_dev2"],
        scale_factor1=psd_params_dict["scale_factor1"],
        scale_factor2=psd_params_dict["scale_factor2"],
    )

    logging.info("Create initial attributes generator")
    initial_attributes_generator = attrsgen.AttrsGeneratorBinWidth(
        radiigen=radii_generator,
        dryradiigen=dryradii_generator,
        xiprobdist=xi_probability_distribution,
        coord3gen=coord3gen,
        coord1gen=coord1gen,
        coord2gen=coord2gen,
    )

    logging.info("Get superdroplets at domain top")
    ### ----- write initial superdroplets binary ----- ###
    with Capturing() as super_top_info:
        number_superdroplets = crdgens.nsupers_at_domain_top(
            grid_filename=grid_file_path,
            constants_filename=constants_file_path,
            nsupers=sd_per_gridbox,
            zlim=zgrid_cloud_base,
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
        nspacedims=1, ngbxs=number_gridboxes_total, maxnsupers=max_number_supers
    )

    editconfigfile.edit_config_params(config_file_path, eurec4a1d_config)

    # --- WRITE THE BINARY FILES ---
    logging.info("Write initial superdroplets binary")
    with Capturing() as super_info:
        try:
            csupers.write_initsuperdrops_binary(
                initsupers_filename=init_superdroplets_file_path,
                initattrsgen=initial_attributes_generator,
                config_filename=config_file_path,
                constants_filename=constants_file_path,
                grid_filename=grid_file_path,
                nsupers=number_superdroplets,
                NUMCONC=0,
            )
        except Exception as e:
            logging.error("Error writing initial superdroplets binary file")
            raise e

    ### ---------------------------------------------------------------- ###
    ### UPDATE THE BOUNDARY CONDITIONS FOR THE CONFIG FILE ###
    ### ---------------------------------------------------------------- ###
    logging.info("Update boundary conditions in config file")
    eurec4a1d_config["boundary_conditions"].update(
        COORD3LIM=float(
            zgrid_cloud_base
        ),  # SDs added to domain with coord3 >= z_boundary_respawn [m]
        newnsupers=sd_per_gridbox,  # number of new super-droplets per gridbox
        MINRADIUS=radius_minimum,  # minimum radius of new super-droplets [m]
        MAXRADIUS=radius_maximum,  # maximum radius of new super-droplets [m]
        NUMCONC_a=psd_params_dict[
            "scale_factor1"
        ],  # number conc. of 1st droplet lognormal dist [m^-3]
        GEOMEAN_a=psd_params_dict[
            "geometric_mean1"
        ],  # geometric mean radius of 1st lognormal dist [m]
        geosigma_a=psd_params_dict[
            "geometric_std_dev1"
        ],  # geometric standard deviation of 1st lognormal dist
        NUMCONC_b=psd_params_dict[
            "scale_factor2"
        ],  # number conc. of 2nd droplet lognormal dist [m^-3]
        GEOMEAN_b=psd_params_dict[
            "geometric_mean2"
        ],  # geometric mean radius of 2nd lognormal dist [m]
        geosigma_b=psd_params_dict[
            "geometric_std_dev2"
        ],  # geometric standard deviation of 2nd lognormal dist
    )
    editconfigfile.edit_config_params(config_file_path, eurec4a1d_config)

    # --- PLOTTING ---

    logging.info("Plot figures")
    fig_dir = individual_output_dir_path / "figures"
    fig_dir.mkdir(exist_ok=True, parents=False)

    gridbox_to_plot = number_gridboxes_total - 1

    isfigures = [True, True]  # booleans for [making, saving] initialisation figures
    ### ----- show (and save) plots of binary file data ----- ###
    with Capturing() as plot_info:
        if isfigures[0]:
            rgrid.plot_gridboxboundaries(
                constants_filename=constants_file_path,
                grid_filename=grid_file_path,
                savefigpath=fig_dir,
                savefig=isfigures[1],
            )
            rthermo.plot_thermodynamics(
                constants_filename=constants_file_path,
                config_filename=config_file_path,
                grid_filename=grid_file_path,
                thermofiles=thermodynamics_file_path,
                savefigpath=fig_dir,
                savefig=isfigures[1],
            )
            rsupers.plot_initGBxs_distribs(
                config_filename=config_file_path,
                constants_filename=constants_file_path,
                initsupers_filename=init_superdroplets_file_path,
                grid_filename=grid_file_path,
                savefigpath=fig_dir,
                savefig=isfigures[1],
                gbxs2plt=gridbox_to_plot,
            )

            plt.close("all")
