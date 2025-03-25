#!/bin/bash
#SBATCH --job-name=e1d_run_CLEO
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=um1487
#SBATCH --output=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/run_CLEO_single/%j_out.log
#SBATCH --error=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/run_CLEO_single/%j_err.log

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
env=/work/um1487/m301096/conda/envs/sdm_pysd_python312/
conda activate ${env}
spack load cmake@3.23.1%gcc
### ---------------------------------------------------- ###

microphysics="null_microphysics"
path2CLEO=${HOME}/CLEO
path2data=${path2CLEO}/data/output_v4.1/
path2build=${path2CLEO}/build_eurec4a1d_openmp/


path2clouddata=${path2data}/${microphysics}/cluster_81

# the following paths will be given by the master submit scrip, which sets the slurm array size in this script too.
echo "init microphysics: ${microphysics}"    # microphysics setup
echo "init path2CLEO: ${path2CLEO}"          # path to the CLEO directory
echo "init path2data: ${path2data}"          # path to the data directory with subdirectories for each microphysics setup
echo "init path2build: ${path2build}"        # path to the build directory

# some example paths which could be used for testing
# path2CLEO=${HOME}/CLEO/
# path2build=${path2CLEO}/build_eurec4a1d/
# path2data=${path2CLEO}/data/test/

# relative paths and names within an individual cloud directory
# individual directory
# | --- config_dir_relative
# |     | --- config_file_name
# | --- dataset_file_relative
config_dir_relative="config"
config_file_relative="${config_dir_relative}/eurec4a1d_config.yaml"
dataset_file_relative="eurec4a1d_sol.zarr"

# initialize
executable_name="eurec4a1d_${microphysics}"
executable2run="${path2build}/examples/eurec4a1d/stationary_${microphysics}/src/${executable_name}"
echo executable_name: ${executable_name}
echo executable2run: ${executable2run}

echo "### ---------------------------------------------------- ###"
### ---------------------------------------------------- ###

### ---------------------------------------------------- ###
# Setup paths to the config file and the dataset file
configfile2run="${path2clouddata}/${config_file_relative}"
dataset2run="${path2clouddata}/${dataset_file_relative}"
# Setup path to the executable
echo "### ---------------------------------------------------- ###"
### ---------------------------------------------------- ###


### ---------------------------------------------------- ###
echo "Validate all paths before running the model"
if [ ! -d "$path2CLEO" ]; then
    echo "Invalid path to CLEO"
    exit 1
elif [ ! -d "$path2build" ]; then
    echo "Invalid path to build"
    exit 1
elif [ ! -d "$path2clouddata" ]; then
    echo "Invalid path to data directory"
    exit 1
elif [ ! -f "$executable2run" ]; then
    echo "Executable not found: ${executable2run}"
    exit 1
elif [ ! -f "$configfile2run" ]; then
    echo "Config file not found: ${configfile2run}"
    exit 1
else
    echo "All paths are valid"
fi
echo "### ---------------------------------------------------- ###"
### ---------------------------------------------------- ###

### ---------------------------------------------------- ###
echo "Delete dataset directory if it exists"
# Check if the directory exists
if [ -d "$dataset2run" ]; then
    echo "Attempting to delete dataset directory: ${dataset2run}"

    # Check for open file descriptors
    if lsof +D "$dataset2run" > /dev/null; then
        echo "Error: Processes are still accessing files in ${dataset2run}. Terminate them before deletion." >&2
        lsof +D "$dataset2run" # Optionally list offending processes
        exit 1
    fi

    # Remove the directory recursively
    rm -rf "$dataset2run"
    if [ $? -ne 0 ]; then
        echo "Error: rm command failed!" >&2
        exit 1
    fi
    echo "Dataset directory deleted successfully."
else
    echo "Directory ${dataset2run} does not exist. No action taken."
fi
echo "### ---------------------------------------------------- ###"
### ---------------------------------------------------- ###


### ---------------------------------------------------- ###
echo "Run the model"

set -e
module purge
spack unload --all

### ------------------ input parameters ---------------- ###
### ----- You need to edit these lines to specify ------ ###
### ----- your build configuration and executables ----- ###
### ---------------------------------------------------- ###
bashsrc=${path2CLEO}/scripts/bash/src
### -------------------- check inputs ------------------ ###

enableyac=false
buildtype="openmp" # as defined by Kokkos configuration; see below
compilername="intel" # as defined by Kokkos configuration; see below
stacksize_limit=204800 # ulimit -s [stacksize_limit] (kB)

export CLEO_PATH2CLEO=${path2CLEO}
export CLEO_BUILDTYPE=${buildtype}
export CLEO_ENABLEYAC=${enableyac}

source ${bashsrc}/check_inputs.sh
check_args_not_empty "${executable2run}" "${configfile2run}" "${CLEO_ENABLEYAC}"
### ---------------------------------------------------- ###


### ----------------- run executable --------------- ###
source ${bashsrc}/runtime_settings.sh ${stacksize_limit}
runcmd="${executable2run} ${configfile2run}"
echo ${runcmd}
eval ${runcmd}
### ---------------------------------------------------- ###
