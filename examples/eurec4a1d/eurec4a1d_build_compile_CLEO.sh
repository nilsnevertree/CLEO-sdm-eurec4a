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
path2eurec4a1d=${path2CLEO}examples/eurec4a1d/


# Use the stationary or evolving version of the model

### ---------- Setup for the EUREC4A1D model ---------- ###

# --- stationary version, with super droplet creation at domain top by boundarz conditions

# NO PHYSICS
path2build=${path2builds}build_eurec4a1D_stationary_no_physics/

# # CONDENSTATION
# path2build=${path2builds}build_eurec4a1D_stationary_condensation/

# # COLLISION AND CONDENSTATION
# path2build=${path2builds}build_eurec4a1D_stationary_collision_condensation/

### ---------------------------------------------------- ###



### ------------------ Load Modules -------------------- ###
cleoenv=/work/mh1126/m300950/cleoenv
python=${cleoenv}/bin/python3
spack load cmake@3.23.1%gcc
module load python3/2022.01-gcc-11.2.0
source activate ${cleoenv}
### ---------------------------------------------------- ###

### -------------------- print inputs ------------------ ###
echo "----- Build and compile CLEO -----"
echo "buildtype:  ${buildtype}"
echo "path2CLEO: ${path2CLEO}"
echo "path2build: ${path2build}"
echo "executables: ${executables}"
echo "---------------------------"
### ---------------------------------------------------- ###

## ---------------------- build CLEO ------------------ ###
${path2CLEO}/scripts/bash/build_cleo.sh ${buildtype} ${path2CLEO} ${path2build}
### ---------------------------------------------------- ###

### --------- compile executable(s) from scratch ---------- ###
cd ${path2build} && make clean

${path2CLEO}/scripts/bash/compile_cleo.sh ${cleoenv} ${buildtype} ${path2build} "${executables}"
## ---------------------------------------------------- ###

### --------- run model through Python script ---------- ###
export OMP_PROC_BIND=spread
export OMP_PLACES=threads


### ---------------------------------------------------- ###

echo "--------------------------------------------"
echo "END RUN"
date
echo "============================================"
