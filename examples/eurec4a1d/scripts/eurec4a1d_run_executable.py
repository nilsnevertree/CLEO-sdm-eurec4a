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


import os
import sys
from pathlib import Path
import yaml
import argparse
import warnings

print(f"Enviroment: {sys.prefix}")

parser = argparse.ArgumentParser()
parser.add_argument("path2CLEO", type=str, help="Path to CLEO")
parser.add_argument("path2build", type=str, help="Path to build")
parser.add_argument(
    "raw_dir_individual",
    type=str,
    help="Path to directory which contains config files and raw data direcotry. Needs to contain 'config/eurec4a_config.yaml'. Output will be stored in /eurec4a1d_sol.zarr.",
)
args = parser.parse_args()

path2CLEO = Path(args.path2CLEO)
path2build = Path(args.path2build)
raw_dir_individual = Path(args.raw_dir_individual)


print(f"path2CLEO: {path2CLEO}")
print(f"path2build: {path2build}")


# Setup paths to the config file and the dataset file
config_dir = raw_dir_individual / "config"
config_file_path = config_dir / "eurec4a1d_config.yaml"
dataset_file_path = raw_dir_individual / "eurec4a1d_sol.zarr"

with open(config_file_path, "r") as f:
    config_dict = yaml.safe_load(f)

if dataset_file_path.is_dir():
    warnings.warn(f"Directory {dataset_file_path} already exists. It will be deleted.")


print("===============================")
print("Executing EUREC4A1D executable")

executable = str(path2build / "examples/eurec4a1d/src/eurec4a1D")
print("Executable: " + executable)
print("Config file: " + str(config_file_path))
print("Dataset file: " + str(dataset_file_path))
print("Try deleting existing dataset file:")
os.system("rm -rf " + str(dataset_file_path))  # delete any existing dataset

# Run the executable
os.chdir(str(path2build))
os.system("pwd")
os.system(executable + " " + str(config_file_path))

print("===============================")
