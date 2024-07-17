#!/bin/bash
#SBATCH --job-name=e1d_create_init_files
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:01:00
#SBATCH --mem=500M
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/create_init_files/%A/%A_%a_out.out
#SBATCH --error=./logfiles/create_init_files/%A/%A_%a_err.out
#SBATCH --array=0-1

### ---------------------------------------------------- ###
### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ----- environment, build type, directories, the ---- ###
### --------- executable(s) to compile and your -------- ###
### --------------  python script to run. -------------- ###
### ---------------------------------------------------- ###

echo "--------------------------------------------"
echo "START RUN"
date
echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "============================================"

echo "init microphysics: ${microphysics}"
echo "init config_directory: ${config_directory}"
echo "init path2CLEO: ${path2CLEO}"
echo "init path2data: ${path2data}"

path2eurec4a1d=${path2CLEO}/examples/eurec4a1d/

# input creation script
pythonscript=${path2eurec4a1d}/scripts/eurec4a1d_stationary.py

# baseline config file
configfile=${path2eurec4a1d}/default_config/eurec4a1d_config_stationary.yaml

### ---------- Setup for the EUREC4A1D model ---------- ###
src_directory=${path2eurec4a1d}/stationary_${setup}/src/

rawdirectory=""
breakup_file="None"

function prepare_microphysics_setup() {
  local setup=$1
  # Update the global variable within the function
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
    breakup_file="${path2eurec4a1d}/src/breakup.yaml"
elif [ "${microphysics}" == "coalbure_condensation_large" ]; then
    prepare_microphysics_setup "${microphysics}"
    breakup_file="${path2eurec4a1d}/stationary_${setup}/src/breakup.yaml"
else
    echo "ERROR: microphysics not found"
    exit 1
fi
### ---------------------------------------------------- ###


### ------------------ Load Modules -------------------- ###
condaenv=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
python=${condaenv}/bin/python3
source activate ${condaenv}
### ---------------------------------------------------- ###


### ---------------------------------------------------- ###
files=($(find ${config_directory} -maxdepth 1 -type f -name 'clusters*.yaml' -printf '%P\n' | sort))
cloud_observation_file=${config_directory}/${files[$SLURM_ARRAY_TASK_ID]}

echo "Number of files: ${#files[@]}"
echo "Current array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Current config file: ${files[${SLURM_ARRAY_TASK_ID}]}"


### -------------------- print inputs ------------------ ###
echo "----- Running Example -----"
echo "path2CLEO: ${path2CLEO}"
echo "pythonscript: ${pythonscript}"
echo "base config file: ${configfile}"
echo "cloud config file: ${cloud_observation_file}"
echo "breakup file: ${breakup_file}"
echo "raw data directory: ${rawdirectory}"
echo "---------------------------"
### ---------------------------------------------------- ###

# make sure paths are directories and executable is a file
if [ ! -d "$path2CLEO" ]; then
    echo "Invalid path to CLEO"
    exit 1
elif [ ! -f "$pythonscript" ]; then
    echo "Invalid path to python script"
    exit 1
elif [ ! -f "$configfile" ]; then
    echo "Invalid path to config file"
    exit 1
elif [ ! -f "$cloud_observation_file" ]; then
    echo "Invalid path to cloud config file"
    exit 1
else
    echo "All paths are valid"
fi


srun ${python} \
    ${pythonscript} \
        ${path2CLEO} \
        ${configfile} \
        ${cloud_observation_file} \
        ${rawdirectory} \
        ${breakup_file} \

echo "--------------------------------------------"
date
echo "END RUN"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
