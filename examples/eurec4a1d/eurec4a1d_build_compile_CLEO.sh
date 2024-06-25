#!/bin/bash
#SBATCH --job-name=eurec4a1d_build_compile_CLEO
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=30G
#SBATCH --time=04:30:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eurec4a1d_build_compile_CLEO.%j_out.out
#SBATCH --error=./logfiles/eurec4a1d_build_compile_CLEO.%j_err.out

### ---------------------------------------------------- ###
### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ----- environment, build type, directories, the ---- ###
### --------- executable(s) to compile and your -------- ###
### --------------  python script to run. -------------- ###
### ---------------------------------------------------- ###

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "START RUN"
date
echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "============================================"

# run parameters
buildtype="cuda"
executables="eurec4a1D"
enableyac=false

# setps to run
build=true
compile=true

# set paths
path2CLEO=${HOME}/CLEO/
path2builds=${path2CLEO}builds/
path2data=${path2CLEO}data/output_v3.1/
path2eurec4a1d=${path2CLEO}examples/eurec4a1d/

# python script to run
pythonscript=${path2eurec4a1d}scripts/eurec4a1d_run_executable.py


# ----- Directory for cloud configuration files ------ #
path2sdmeurec4a=${HOME}/repositories/sdm-eurec4a/
cloud_config_directory=${path2sdmeurec4a}data/model/input/output_v3.0/
# ---------------------------------------------------- #



### ---------- Setup for the EUREC4A1D model ---------- ###
# Use the stationary setup of the model

# NO PHYSICS
path2build=${path2builds}build_eurec4a1D_stationary_no_physics/
rawdirectory=${path2data}stationary_no_physics/

# # CONDENSTATION
# path2build=${path2builds}build_eurec4a1D_stationary_condensation/
# rawdirectory=${path2data}stationary_condensation/

# # COLLISION AND CONDENSTATION
# path2build=${path2builds}build_eurec4a1D_stationary_collision_condensation/
# rawdirectory=${path2data}stationary_collision_condensation/

### ---------------------------------------------------- ###



### ------------------ Load Modules -------------------- ###
cleoenv=/work/mh1126/m300950/cleoenv
python=${cleoenv}/bin/python3
yacyaxtroot=/work/mh1126/m300950/yac
spack load cmake@3.23.1%gcc
module load python3/2022.01-gcc-11.2.0
source activate ${cleoenv}
### ---------------------------------------------------- ###

### -------------------- print inputs ------------------ ###
echo "----- Build and compile CLEO -----"
echo "buildtype:  ${buildtype}"
echo "path2CLEO: ${path2CLEO}"
echo "path2build: ${path2build}"
echo "enableyac: ${enableyac}"
echo "executables: ${executables}"
echo "pythonscript: ${pythonscript}"
echo "rawdirectory: ${rawdirectory}"
echo "cloud_config_directory: ${cloud_config_directory}"
echo "---------------------------"
### --------------------------------------------------- ###

## ---------------------- build CLEO ------------------ ###
if [ "$build" = true ]; then
    ${path2CLEO}/scripts/bash/build_cleo.sh ${buildtype} ${path2CLEO} ${path2build}
fi
### ---------------------------------------------------- ###

### --------- compile executable(s) from scratch ------- ###
if [ "$compile" = true ]; then
    cd ${path2build} && make clean
    ${path2CLEO}/scripts/bash/compile_cleo.sh ${cleoenv} ${buildtype} ${path2build} "${executables}"
fi
### ---------------------------------------------------- ###

echo "--------------------------------------------"
date
echo "END RUN"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
