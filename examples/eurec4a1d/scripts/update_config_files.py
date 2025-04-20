"""
This script is used to run the EUREC4A1D executable. It is called by the
`eurec4a1d_run_executable.sh` script. The script takes the following arguments:

1. path2CLEO: Path to the CLEO repository
2. path2build: Path to the build directory
3. raw_dir_individual: Path to the directory which contains the config files and raw data directory.
   Needs to contain 'config/eurec4a_config.yaml'. Output will be stored in /eurec4a1d_sol.zarr.
   raw_dir_individual
    ├── config
    │   └── eurec4a1d_config.yaml   <- NEEDS TO EXIST
    └── eurec4a1d_sol.zarr          <- will be created by the executable
"""

# %%

import sys
import secrets
from pathlib import Path
import ruamel.yaml
import logging
import datetime
import numpy as np
from mpi4py import MPI

yaml = ruamel.yaml.YAML()
# logging configure
logging.basicConfig(level=logging.INFO)


path2CLEO = Path(__file__).resolve().parents[3]

path2build = path2CLEO / "build_eurec4a1d"
path2eurec4a1d = path2CLEO / "examples/eurec4a1d"
# === logging ===
# create log file

time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

# create shared logging directory
try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # [0,1,2,3,4,5,6,7,8,9]
    npro = comm.Get_size()  # 10
except Exception:
    raise ImportError("MPI4PY not possible!")

# create shared logging directory
if rank == 0:
    # Generate a shared directory name based on UTC time and random hex
    time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    random_hex = secrets.token_hex(4)
    log_dir = (
        path2eurec4a1d / "logfiles" / f"update_config_files/{time_str}-{random_hex}"
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
logging.info(f"Start with rank {rank} of {npro}")

logging.info(f"Enviroment: {sys.prefix}")

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "data_dir",
#     type=str,
#     help="Path to directory which contains the subdirectories which contain the config files.",
# )
# parser.add_argument(
#     "config_file_relative_path",
#     type=str,
#     default="config/eurec4a1d_config.yaml",
#     help="Relative path to the config file from the data_dir.",
# )
# parser.add_argument(
#     "sub_dir_pattern",
#     type=str,
#     default="cluster*",
#     help="Pattern to match the subdirectories in the data_dir.",
# )
# args = parser.parse_args()
# data_dir = args.data_dir
# config_file_relative_path = args.config_file_relative_path
# sub_dir_pattern = args.sub_dir_pattern


data_dir = path2CLEO / "data/output_v4.1/null_microphysics"
config_file_relative_path = "config/eurec4a1d_config.yaml"
sub_dir_pattern = "cluster*"


logging.info(f"Data directory: {data_dir}")
logging.info(f"Config file relative path: {config_file_relative_path}")
logging.info(f"Sub directory pattern: {sub_dir_pattern}")

data_dir_path = Path(data_dir)
config_file_relative_path = Path(config_file_relative_path)
sub_dir_pattern = sub_dir_pattern

all_config_paths = (
    np.array(sorted(list(data_dir_path.glob(f"{sub_dir_pattern}"))))
    / config_file_relative_path
)

rank_config_paths = np.array_split(all_config_paths, npro)[rank]

for step, config_path in enumerate(rank_config_paths):
    logging.info(f"Core {rank +1} Step {step+1}/{len(rank_config_paths)}")
    logging.info(f"Updating config file: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.load(f)
    # data["timesteps"] = dict(
    #     CONDTSTEP=0.05,
    #     COLLTSTEP=2,
    #     MOTIONTSTEP=2,
    #     COUPLTSTEP=3600,
    #     OBSTSTEP=2,
    #     T_END=3600,
    # )
    # data["microphysics"]["condensation"] = dict(
    #     do_alter_thermo=False,
    #     maxniters=100,
    #     MINSUBTSTEP=0.01,
    #     rtol=1.0,
    #     atol=0.1,
    # )
    try:
        data["kokkos_settings"] = dict(
            num_threads=128
        )  # number of threads for host parallel backend
    except Exception:
        logging.info("kokkos_settings already exists")
        data.insert(
            1,
            "kokkos_settings",
            dict(num_threads=128),
            comment="number of threads for host parallel backend",
        )

    with open(config_path, "w") as file:
        yaml.dump(data, file)
