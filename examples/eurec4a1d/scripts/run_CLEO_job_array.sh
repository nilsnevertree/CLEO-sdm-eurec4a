#!/bin/bash
#SBATCH --job-name=e1d_run_CLEO
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/run_CLEO/%A/%A_%a_out.out
#SBATCH --error=./logfiles/run_CLEO/%A/%A_%a_err.out
#SBATCH --array=0-7

### ---------------------------------------------------- ###
### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ----- environment, build type, directories, the ---- ###
### --------- exec(s) to compile and your -------- ###
### --------------  python script to run. -------------- ###
### ---------------------------------------------------- ###

# Ensure script exits on any error
echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "date: $(date)"
echo "============================================"

### ------------------ Load Modules -------------------- ###
source ${HOME}/.bashrc
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
conda activate ${env}
spack load cmake@3.23.1%gcc
### ---------------------------------------------------- ###


echo "init microphysics: ${microphysics}"
echo "init config_directory: ${config_directory}"
echo "init path2CLEO: ${path2CLEO}"
echo "init path2data: ${path2data}"
echo "init path2build: ${path2build}"

run=true

# set paths
# path2CLEO=${HOME}/CLEO/
# path2build=${path2CLEO}/build_eurec4a1d/
# path2data=${path2CLEO}/data/test/
path2eurec4a1d=${path2CLEO}/examples/eurec4a1d/


### ---------- Setup for the EUREC4A1D model ---------- ###

exec=""
path2exec=""
rawdirectory=""

# function to set the rawdirectory and the exec and path2exec
function prepare_microphysics_setup() {
  local setup=$1
  exec="eurec4a1D_${setup}"
  path2exec="${path2build}/examples/eurec4a1d/stationary_${setup}/src/${exec}"
  rawdirectory="${path2data}/${setup}/"
}

# Your existing conditional logic
if [ "${microphysics}" == "null_microphysics" ]; then
    prepare_microphysics_setup "${microphysics}"
elif [ "${microphysics}" == "condensation" ]; then
    prepare_microphysics_setup "${microphysics}"
elif [ "${microphysics}" == "collision_condensation" ]; then
    prepare_microphysics_setup "${microphysics}"
elif [ "${microphysics}" == "coalbure_condensation_small" ]; then
    prepare_microphysics_setup "${microphysics}"
elif [ "${microphysics}" == "coalbure_condensation_large" ]; then
    prepare_microphysics_setup "${microphysics}"
else
    echo "ERROR: microphysics not found"
    exit 1
fi
### ---------------------------------------------------- ###

### ---------------------------------------------------- ###
# Setup paths depending on current array task ID
# directories=(${rawdirectory}/${subdir_pattern}*)
# IMPORANT: The directories must be sorted to match the array task ID
directories=($(find ${rawdirectory} -maxdepth 1 -type d -name 'cluster*' -printf '%P\n' | sort))

#echo "Directories: ${directories[@]}"
echo "Number of directories: ${#directories[@]}"
echo "Current array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Current directory: ${directories[${SLURM_ARRAY_TASK_ID}]}"

path2inddir=${rawdirectory}/${directories[${SLURM_ARRAY_TASK_ID}]}


config_dir_name="config"
config_file_name="eurec4a1d_config.yaml"
dataset_name="eurec4a1d_sol.zarr"

# Setup paths to the config file and the dataset file
config_file_path="${path2inddir}/${config_dir_name}/${config_file_name}"
dataset_path="${path2inddir}/${dataset_name}"
# Setup path to the executable
### ---------------------------------------------------- ###



### -------------------- print inputs ------------------ ###
echo "============================================"
echo -e "buildtype: \t${buildtype}"
echo -e "path2CLEO: \t${path2CLEO}"
echo -e "path2build: \t${path2build}"
echo -e "enableyac: \t${enableyac}"
echo "--------------------------------------------"
echo -e "microphysics: \t${microphysics}"
echo -e "exec: \t$(basename ${path2exec})"
echo -e "path2exec: \t${path2exec}"
echo "--------------------------------------------"
echo -e "base directory: \t${path2inddir}"
echo -e "config file: \t\t${config_file_path}"
echo -e "dataset file: \t\t${dataset_path}"
echo "============================================"
### --------------------------------------------------- ###


# make sure paths are directories and executable is a file
if [ ! -d "$path2CLEO" ]; then
    echo "Invalid path to CLEO"
    exit 1
elif [ ! -d "$path2build" ]; then
    echo "Invalid path to build"
    exit 1
elif [ ! -d "$path2inddir" ]; then
    echo "Invalid path to data directory"
    exit 1
elif [ ! -f "$path2exec" ]; then
    echo "Executable not found: ${path2exec}"
    exit 1
elif [ ! -f "$config_file_path" ]; then
    echo "Config file not found: ${config_file_path}"
    exit 1
else
    echo "All paths are valid"
fi

# Check if the directory exists
if [ -d "$dataset_path" ]; then
    echo "Attempting to delete dataset directory: ${dataset_path}"

    # Check for open file descriptors
    if lsof +D "$dataset_path" > /dev/null; then
        echo "Error: Processes are still accessing files in ${dataset_path}. Terminate them before deletion." >&2
        lsof +D "$dataset_path" # Optionally list offending processes
        exit 1
    fi

    # Remove the directory recursively
    rm -rf "$dataset_path"
    if [ $? -ne 0 ]; then
        echo "Error: rm command failed!" >&2
        exit 1
    fi
    echo "Dataset directory deleted successfully."
else
    echo "Directory ${dataset_path} does not exist. No action taken."
fi

echo "============================================"

### --------- run model through Python script ---------- ###
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Change to the build directory
cd ${path2build}
echo "Current directory: $(pwd)"

echo "============================================"
echo "Run CLEO in ${directory_individual}"
# Execute the executable
echo "Executing executable ${executable} with config file ${config_file_path}"
${path2exec} ${config_file_path}

echo "============================================"
date
echo "END RUN"
