#!/bin/bash
#SBATCH --job-name=e1d_build_compile_CLEO
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=30G
#SBATCH --time=04:30:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/build_compile_CLEO/%j_out.out
#SBATCH --error=./logfiles/build_compile_CLEO/%j_err.out

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
executables="eurec4a1D_null_microphysics eurec4a1D_condensation eurec4a1D_collision_condensation eurec4a1D_coalbure_condensation_small eurec4a1D_coalbure_condensation_large eurec4a1D_coalbure_condensation_cke"
enableyac=false

# setps to run
build=true
compile=true

# set paths
path2CLEO=${HOME}/CLEO/
path2build=${path2CLEO}/build_eurec4a1d/

### ------------------ Load Modules -------------------- ###
cleoenv=/work/mh1126/m300950/cleoenv
python=${cleoenv}/bin/python3
yacyaxtroot=/work/mh1126/m300950/yac
spack load cmake@3.23.1%gcc
module load python3/2022.01-gcc-11.2.0
source activate ${cleoenv}
### ---------------------------------------------------- ###

### -------------------- print inputs ------------------ ###
echo "============================================"
echo -e "buildtype: \t${buildtype}"
echo -e "path2CLEO: \t${path2CLEO}"
echo -e "path2build: \t${path2build}"
echo -e "enableyac: \t${enableyac}"
echo -e "executable: \t${executable}"
echo "============================================"
### --------------------------------------------------- ###

## ---------------------- build CLEO ------------------ ###
if [ "$build" = true ]; then
    echo "Build CLEO"
    ${path2CLEO}/scripts/bash/build_cleo.sh ${buildtype} ${path2CLEO} ${path2build}
    echo "============================================"
fi
### ---------------------------------------------------- ###

### --------- compile executable(s) from scratch ------- ###
if [ "$compile" = true ]; then
    echo "Compile CLEO"
    cd ${path2build} && make clean
    ${path2CLEO}/scripts/bash/compile_cleo.sh ${cleoenv} ${buildtype} ${path2build} "${executables}"
    echo "============================================"
fi
### ---------------------------------------------------- ###

echo "--------------------------------------------"
date
echo "END RUN"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
