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
from pathlib import Path
import yaml
import argparse

from pySD import editconfigfile


print(f"Enviroment: {sys.prefix}")

parser = argparse.ArgumentParser()
parser.add_argument(
    "raw_dir_individual",
    type=str,
    help="Path to directory which contains config files and raw data direcotry. Needs to contain 'config/eurec4a_config.yaml'. Output will be stored in /eurec4a1d_sol.zarr.",
)
args = parser.parse_args()

raw_dir_individual = Path(args.raw_dir_individual)


# Setup paths to the config file and the dataset file
config_dir = raw_dir_individual / "config"
config_file_path = config_dir / "eurec4a1d_config.yaml"

print(f"Config file path: {config_file_path}")

with open(str(config_file_path), "r") as f:
    config_dict = yaml.safe_load(f)

config_dict["microphysics"]["condensation"] = dict(
    do_alter_thermo=False,
    maxniters=100,
    MINSUBTSTEP=0.01,
    rtol=0.03,
    atol=0.03,
)

editconfigfile.edit_config_params(str(config_file_path), config_dict)

print("Sucessfully updated config file.")
