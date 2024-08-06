#!/bin/bash
#SBATCH --job-name=CLEO_eurec4a1d-%a
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3G
#SBATCH --time=00:15:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/execution/CLEO_eurec4a1d-%A_%a_out.out
#SBATCH --error=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/execution/CLEO_eurec4a1d-%A_%a_err.out


# This script is used to run the EUREC4A1D executable.
# It should be executed with the sbatch --array this_scrpt path2CLEO path2build path2inddir executable

# The script takes the following arguments:

# 1. path2CLEO: Path to the CLEO repository
# 2. path2build: Path to the build directory
# 3. path2inddir: Path to the directory which contains the config files and raw data directory.
#    Needs to contain '/config/eurec4a_config.yaml'.
#    Output will be stored in /eurec4a1d_sol.zarr.
#    path2inddir
#     ├── config
#     │   └── eurec4a1d_config.yaml   <- NEEDS TO EXIST
#     └── eurec4a1d_sol.zarr          <- will be created by the executable
# 4. executable: Path to the executable to run. It is assumed, that it is within the build directory.

path2CLEO=$1
path2build=$2
path2inddir=$3 # The path to the raw data directory
executable=$4
# make sure paths are directories and executable is a file
if [ ! -d "$path2CLEO" ]; then
    echo " (1) Invalid path to CLEO"
    exit 1
fi
if [ ! -d "$path2build" ]; then
    echo " (2) Invalid path to build"
    exit 1
fi
if [ ! -d "$path2inddir" ]; then
    echo "(3) Invalid path to data directory"
    exit 1
fi
if [ ! -f "$executable" ]; then
    echo "(4) Executable not found: ${executable}"
    exit 1
fi

config_dir_name="config"
config_file_name="eurec4a1d_config.yaml"
dataset_name="eurec4a1d_sol.zarr"

# Setup paths to the config file and the dataset file
config_dir="${path2inddir}/${config_dir_name}"
config_file_path="${config_dir}/${config_file_name}"
dataset_path="${path2inddir}/${dataset_name}"
# Setup path to the executable

if [ ! -f "$config_file_path" ]; then
    echo "Config file not found: ${config_file_path}"
    exit 1
fi


echo -e "path2CLEO: \t\t${path2CLEO}"
echo -e "path2build: \t\t${path2build}"
echo -e "Exec. name: \t\t$(basename ${executable})"
echo -e "Exec. path: \t\t${executable}"
echo -e "Base directory: \t${path2inddir}"
echo -e "Config directory: \t${config_dir}"
echo -e "Config file: \t\t${config_file_path}"
echo -e "Dataset file: \t\t${dataset_path}"

# Check if the directory exists
if [ -d "$dataset_path" ]; then
    echo "Attempt to delet existing dataset file: ${dataset_path}"
    rm -rf ${dataset_path} & echo "Dataset file deleted"
fi

# Change to the build directory
cd ${path2build}
echo "Current directory: $(pwd)"

# Execute the executable
echo "Executing executable ${executable} with config file ${config_file_path}"
srun ${executable} ${config_file_path}
