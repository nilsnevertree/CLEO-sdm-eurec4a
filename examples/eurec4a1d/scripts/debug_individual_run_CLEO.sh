#!/bin/bash
#SBATCH --job-name=e1d_run_CLEO
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --time=00:15:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
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

echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "date: $(date)"
echo "============================================"


path2CLEO=${HOME}/CLEO/
path2build=${path2CLEO}/build_eurec4a1d/
path2eurec4a1d=${path2CLEO}/examples/eurec4a1d/
microphysics=condensation
path2data=${path2CLEO}/data/debug_output/

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

rawdirectory=${path2data}
### ---------------------------------------------------- ###

### ---------------------------------------------------- ###


path2inddir=${rawdirectory}/cluster_110


config_dir_name="config"
config_file_name="eurec4a1d_config.yaml"
dataset_name="eurec4a1d_sol.zarr"

# Setup paths to the config file and the dataset file
config_file_path="${path2inddir}/${config_dir_name}/${config_file_name}"
dataset_path="${path2inddir}/${dataset_name}"
# Setup path to the executable
### ---------------------------------------------------- ###


### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
mamba activate ${env}
spack load cmake@3.23.1%gcc
# module load python3/2022.01-gcc-11.2.0
source activate ${cleoenv}
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
    echo "Attempt to delet existing dataset file: ${dataset_path}"
    rm -rf ${dataset_path} & echo "Dataset file deleted"
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
