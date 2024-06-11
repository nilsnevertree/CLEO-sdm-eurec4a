#!/bin/bash
#SBATCH --job-name=eurec4a1d_run_executable_CLEO
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=30G
#SBATCH --time=01:30:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eurec4a1d_run_executable_CLEO.%j_out.out
#SBATCH --error=./logfiles/eurec4a1d_run_executable_CLEO.%j_err.out

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

buildtype="cuda"
executables="eurec4a1D"

path2CLEO=${HOME}/CLEO/
path2builds=${path2CLEO}builds/
path2data=${path2CLEO}data/output_v1.0/
path2eurec4a1d=${path2CLEO}examples/eurec4a1d/

# cloud type
path2sdmeurec4a=${HOME}/repositories/sdm-eurec4a/
cloud_config_directory=${path2sdmeurec4a}data/model/input/new_subset/
# cloud_observation_configfile=${path2sdmeurec4a}data/model/input/new/clusters_18.yaml
# cloud_observation_configfile=${cloud_config_directory}clusters_18.yaml

# Use the stationary or evolving version of the model

### ---------- Setup for the EUREC4A1D model ---------- ###

# --- stationary version, with super droplet creation at domain top by boundarz conditions

# NO PHYSICS
path2build=${path2builds}build_eurec4a1D_stationary_no_physics/
rawdirectory=${path2data}stationary_no_physics/

# # CONDENSTATION
# path2build=${path2builds}build_eurec4a1D_stationary_condensation_v2/
# rawdirectory=${path2data}stationary_condensation/



pythonscript=${path2eurec4a1d}scripts/eurec4a1d_run_executable.py

### ---------------------------------------------------- ###



### ------------------ Load Modules -------------------- ###
cleoenv=/work/mh1126/m300950/cleoenv
python=${cleoenv}/bin/python3
spack load cmake@3.23.1%gcc
module load python3/2022.01-gcc-11.2.0
source activate ${cleoenv}
### ---------------------------------------------------- ###

### -------------------- print inputs ------------------ ###
echo "----- Running Example -----"
echo "buildtype:  ${buildtype}"
echo "path2CLEO: ${path2CLEO}"
echo "path2build: ${path2build}"
echo "executables: ${executables}"
echo "pythonscript: ${pythonscript}"
echo "---------------------------"
### ---------------------------------------------------- ###

# ## ---------------------- build CLEO ------------------ ###
# ${path2CLEO}/scripts/bash/build_cleo.sh ${buildtype} ${path2CLEO} ${path2build}
# ### ---------------------------------------------------- ###

# ### --------- compile executable(s) from scratch ---------- ###
# cd ${path2build} && make clean

# ${path2CLEO}/scripts/bash/compile_cleo.sh ${cleoenv} ${buildtype} ${path2build} "${executables}"
# ### ---------------------------------------------------- ###

### --------- run model through Python script ---------- ###
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

for cloud_configfile in ${cloud_config_directory}/*.yaml; do
    echo "::::::::::::::::::::::::::::::::::::::::::::"
    echo "============================================"
    echo "RUNNING CLEO EXECUTABLE"
    echo "with ${cloud_configfile}"
    echo "============================================"

    script_args="${cloud_configfile} ${rawdirectory}"
    {
        ${python}  ${pythonscript} ${path2CLEO} ${path2build} ${script_args}
    } || {
        echo "============================================"
        echo "ERROR: in ${cloud_configfile}"
        echo "============================================"
    }

done


### ---------------------------------------------------- ###

echo "--------------------------------------------"
echo "END RUN"
date
echo "============================================"
