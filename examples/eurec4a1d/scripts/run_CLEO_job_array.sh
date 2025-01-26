#!/bin/bash
#SBATCH --job-name=e1d_run_CLEO
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:25:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/run_CLEO_openmp/%A/%A_%a_out.out
#SBATCH --error=./logfiles/run_CLEO_openmp/%A/%A_%a_err.out
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

### ------------------ Load Modules -------------------- ###
source ${HOME}/.bashrc
### ---------------------------------------------------- ###

# the following paths will be given by the master submit scrip, which sets the slurm array size in this script too.
echo "init microphysics: ${microphysics}"    # microphysics setup
echo "init path2CLEO: ${path2CLEO}"          # path to the CLEO directory
echo "init path2data: ${path2data}"          # path to the data directory with subdirectories for each microphysics setup
echo "init path2build: ${path2build}"        # path to the build directory

action='run'
path2yac=/work/bm1183/m300950/yacyaxt


# some example paths which could be used for testing
# path2CLEO=${HOME}/CLEO/
# path2build=${path2CLEO}/build_eurec4a1d/
# path2data=${path2CLEO}/data/test/

# relative paths and names within an individual cloud directory
# individual directory
# | --- config_dir_name
# |     | --- config_file_name
# | --- dataset_name
config_dir_name="config"
config_file_name="eurec4a1d_config.yaml"
relative_config_path="${config_dir_name}/${config_file_name}"
dataset_name="eurec4a1d_sol.zarr"


### ---------- Setup for the EUREC4A1D model ---------- ###

# initialize
exec=""                         # executable name
path2exec=""                    # path to the executable
path2microphysics_data=""       # path to the directory which contains the subdirectories for each cluster

# function to set the path2microphysics_data and the exec and path2exec
function prepare_microphysics_setup() {
  local setup=$1
  # the executable name is given by the microphysics setup
  exec="eurec4a1D_${setup}"
  # the executable path lies within the build directory and the eurec4a1d directory
  path2exec="${path2build}/examples/eurec4a1d/stationary_${setup}/src/${exec}"
    # the path to the data directory is given by the path2data and the microphysics setup
  path2microphysics_data="${path2data}/${setup}/"
}

# Create the microphysics setup
prepare_microphysics_setup "${microphysics}"
### ---------------------------------------------------- ###

### ---------------------------------------------------- ###
# Setup paths depending on current array task ID
# directories=(${path2microphysics_data}/${subdir_pattern}*)
# IMPORANT: The directories must be sorted to match the array task ID
directories=($(find ${path2microphysics_data} -maxdepth 1 -type d -name 'cluster*' -printf '%P\n' | sort))
current_directory=${directories[${SLURM_ARRAY_TASK_ID}]}
#echo "Directories: ${directories[@]}"
echo "Number of directories: ${#directories[@]}"
echo "Current array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Current directory: ${current_directory}"

path2inddir=${path2microphysics_data}/${current_directory}


# Setup paths to the config file and the dataset file
config_file_path="${path2inddir}/${relative_config_path}"
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


# Validate all paths before running the model
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


# ------------- RUN --------------------- #

if [ "${action}" == "run" ]
then
    cd ${path2build}

    set -e
    module purge
    spack unload --all
    # note version of python must match the YAC python bindings (e.g. module load python3/2022.01-gcc-11.2.0)
    module load openmpi/4.1.2-gcc-11.2.0 # same mpi as loaded for the build
    ### ----------------- load Python ------------------------ ###
    spack load python@3.9.9%gcc@=11.2.0/fwv
    spack load py-cython@0.29.33%gcc@=11.2.0/j7b4fa
    spack load py-mpi4py@3.1.2%gcc@=11.2.0/hdi5yl6
    ### ------------------------------------------------------ ###

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/sw/spack-levante/libfyaml-0.7.12-fvbhgo/lib
    export PYTHONPATH=${PYTHONPATH}:${path2yac}/yac/python # path to python bindings

    export OMP_PROC_BIND=spread
    export OMP_PLACES=threads


    ### ---------------------------------------------------- ###
    ### ---------------------------------------------------- ###
    ### ---------------------------------------------------- ###

    ### ---------- build, compile and run example ---------- ###
    ${path2CLEO}/examples/run_example.sh \
    ${buildtype} ${path2CLEO} ${path2build} ${enableyac} \
    "${executables}" ${pythonscript} "${script_args}"
    ### ---------------------------------------------------- ###

    buildtype="openmp"
    enableyac=false

    path2CLEO=${path2CLEO}
    path2build=${path2build}

    cleoenv=/work/bm1183/m300950/bin/envs/cleoenv
    python=${cleoenv}/bin/python3
    enabledebug=false
    make_clean=false
    yacyaxtroot=/work/bm1183/m300950/yacyaxt
    stacksize_limit=204800 # ulimit -s [stacksize_limit] (kB)

    if [ "${buildtype}" == "cuda" ]
    then
    compilername=gcc
    else
    compilername=intel
    fi
    ### ---------------------------------------------------- ###
    ### ---------------------------------------------------- ###
    ### ---------------------------------------------------- ###

    ### -------------------- print inputs ------------------ ###
    echo "----- Running Example -----"
    echo "buildtype = ${buildtype}"
    echo "compilername = ${compilername}"
    echo "path2CLEO = ${path2CLEO}"
    echo "path2build = ${path2build}"
    echo "enableyac = ${enableyac}"
    echo "---------------------------"
    ### ---------------------------------------------------- ###


    ### --------- run model  ---------- ###
    export CLEO_PATH2CLEO=${path2CLEO}
    export CLEO_BUILDTYPE=${buildtype}
    export CLEO_ENABLEYAC=${enableyac}
    source ${path2CLEO}/scripts/bash/src/runtime_settings.sh ${stacksize_limit}





    executable2run=${path2exec}
    configfile=${config_file_path}
    stacksize_limit=${stacksize_limit} # kB
    bashsrc=${CLEO_PATH2CLEO}/scripts/bash/src
    ### -------------------- check inputs ------------------ ###
    source ${bashsrc}/check_inputs.sh
    check_args_not_empty "${executable2run}" "${configfile}" "${CLEO_ENABLEYAC}"
    ### ---------------------------------------------------- ###

    ### ----------------- run executable --------------- ###
    source ${bashsrc}/runtime_settings.sh ${stacksize_limit}
    runcmd="${executable2run} ${configfile}"
    echo ${runcmd}
    eval ${runcmd}
    ### ---------------------------------------------------- ###

echo "============================================"
date
echo "END RUN"
