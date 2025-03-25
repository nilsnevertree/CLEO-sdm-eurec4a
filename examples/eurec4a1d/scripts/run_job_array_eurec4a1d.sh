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
#SBATCH --output=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/run_CLEO/%A/%A_%a_out.out
#SBATCH --error=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/run_CLEO/%A/%A_%a_err.out
#SBATCH --array=0-127

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

set -e
module purge
spack unload --all



# env=/work/um1487/m301096/conda/envs/sdm_pysd_python312/
# conda activate ${env}
### ---------------------------------------------------- ###

# the following paths will be given by the master submit scrip, which sets the slurm array size in this script too.
echo "----------------------------------------------------"
echo "EUREC4A1D_MICROPHYSICS: ${EUREC4A1D_MICROPHYSICS}"
echo "EUREC4A1D_PATH2DATA: ${EUREC4A1D_PATH2DATA}"
echo "EUREC4A1D_SUBDIR_PATTERN: ${EUREC4A1D_SUBDIR_PATTERN}"
echo "----------------------------------------------------"
echo "CLEO_PATH2CLEO: ${CLEO_PATH2CLEO}"
echo "CLEO_PATH2BUILD: ${CLEO_PATH2BUILD}"
echo "CLEO_STACKSIZE_LIMIT: ${CLEO_STACKSIZE_LIMIT}"
echo "CLEO_ENABLEYAC: ${CLEO_ENABLEYAC}"
echo "CLEO_RUN_EXECUTABLE: ${CLEO_RUN_EXECUTABLE}"
echo "----------------------------------------------------"

### ------------------ Load Modules -------------------- ###
cleo_bashsrc=${CLEO_PATH2CLEO}/scripts/levante/bash/src
local_bashsrc="${HOME}/.bashrc"
source ${local_bashsrc}
source ${cleo_bashsrc}/check_inputs.sh

### -------------------- check inputs ------------------- ###
check_args_not_empty "${EUREC4A1D_MICROPHYSICS}"  "${EUREC4A1D_PATH2DATA}" "${EUREC4A1D_SUBDIR_PATTERN}"
check_args_not_empty "${CLEO_PATH2CLEO}" "${CLEO_PATH2BUILD}" "${CLEO_STACKSIZE_LIMIT}" "${CLEO_ENABLEYAC}" "${CLEO_RUN_EXECUTABLE}"

### ---------- GET CLOUD DIR FOR THIS SLURM_ARRAY_TASK_ID ---------------- ###
microphysics_data_dir=${EUREC4A1D_PATH2DATA}/${EUREC4A1D_MICROPHYSICS}

# find all subdirectories directories with the pattern
directories=($(find ${microphysics_data_dir} -maxdepth 1 -type d -name ${EUREC4A1D_SUBDIR_PATTERN} -printf '%P\n' | sort))
current_directory=${directories[${SLURM_ARRAY_TASK_ID}]}
path2clouddata=${microphysics_data_dir}/${current_directory} # the path to the current cloud directory

check_args_not_empty "${path2clouddata}"

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total number of dirs: ${#directories[@]}"
echo "Current dir name: ${current_directory}"
echo "Current dir path: ${path2clouddata}"
echo "----------------------------------------------------"



# relative paths and names within an individual cloud directory
# individual directory
# | --- config_dir_relative
# |     | --- config_file_name
# | --- dataset_file_relative
config_dir_relative="config"
config_file_relative="${config_dir_relative}/eurec4a1d_config.yaml"
dataset_file_relative="eurec4a1d_sol.zarr"


### ---------- Setup for the EUREC4A1D model ---------- ###
executable2run="${CLEO_RUN_EXECUTABLE}"
configfile2run="${path2clouddata}/${config_file_relative}"
dataset2run="${path2clouddata}/${dataset_file_relative}"

check_args_not_empty "${executable2run}" "${configfile2run}" "${dataset2run}"


echo "executable_name: ${executable_name}"
echo "executable2run: ${executable2run}"
echo "configfile2run: ${configfile2run}"
echo "dataset2run: ${dataset2run}"
echo "----------------------------------------------------"
### ---------------------------------------------------- ###


### ---------------------------------------------------- ###
echo "Validate all paths before running the model"
if [ ! -d "$CLEO_PATH2CLEO" ]; then
    echo "Invalid path to CLEO"
    exit 1
elif [ ! -d "$CLEO_PATH2BUILD" ]; then
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
echo "----------------------------------------------------"
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
echo "----------------------------------------------------"
### ---------------------------------------------------- ###


### ----------------- run executable --------------- ###
echo "Execute CLEO"
echo "date: $(date)"
# create a runtime settings file
source ${cleo_bashsrc}/runtime_settings.sh ${CLEO_STACKSIZE_LIMIT}
runcmd="${executable2run} ${configfile2run}"
echo ${runcmd}
echo "===================================================="
eval ${runcmd}
echo "===================================================="
### ---------------------------------------------------- ###
echo "FINISHED"
echo "date: $(date)"
echo "----------------------------------------------------"
