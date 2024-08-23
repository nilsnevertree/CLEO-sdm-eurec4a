#!/bin/bash
#SBATCH --job-name=eurec4a1d_run_executable_CLEO
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=30G
#SBATCH --time=04:30:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eurec4a1d_run_executable_CLEO.%j_out.out
#SBATCH --error=./logfiles/eurec4a1d_run_executable_CLEO.%j_err.out

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

# run parameters
buildtype="cuda"
executables="eurec4a1D_null_microphysics eurec4a1D_condensation eurec4a1D_collision_condensation"
enableyac=false

# setps to run
build=false
compile=false
run=true

# set paths
path2CLEO=${HOME}/CLEO/
path2build=${path2CLEO}/build_eurec4a1d/
path2data=${path2CLEO}/data/output_new_build/
path2eurec4a1d=${path2CLEO}/examples/eurec4a1d/
subdir_pattern=clusters_

# bash script to run
run_script=${path2eurec4a1d}/scripts/execute_single_run.sh

### ---------- Setup for the EUREC4A1D model ---------- ###
setup="null_microphysics"
# setup="condensation"
# setup="collision_condensation"
exec="eurec4a1D_${setup}"
path2exec=${path2build}/examples/eurec4a1d/stationary_${setup}/src/${exec}
rawdirectory=${path2data}/${setup}/
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
echo "============================================"
echo -e "buildtype: \t${buildtype}"
echo -e "path2CLEO: \t${path2CLEO}"
echo -e "path2build: \t${path2build}"
echo -e "enableyac: \t${enableyac}"
echo -e "setup: \t${setup}"
echo -e "exec: \t${exec}"
echo -e "path2exec: \t${path2exec}"
echo -e "run_script: \t${run_script}"
echo "============================================"
### --------------------------------------------------- ###

## ---------------------- build CLEO ------------------ ###
if [ "$build" = true ]; then
    echo "Build CLEO"
    ${path2CLEO}/scripts/bash/build_cleo.sh ${buildtype} ${path2CLEO} ${path2build}
    echo "============================================"
fi
### ---------------------------------------------------- ###

### --------- compile exec(s) from scratch ------- ###
if [ "$compile" = true ]; then
    echo "Compile CLEO"
    cd ${path2build} && make clean
    ${path2CLEO}/scripts/bash/compile_cleo.sh ${cleoenv} ${buildtype} ${path2build} "${executables}"
    echo "============================================"
fi
### ---------------------------------------------------- ###

### --------- run model through Python script ---------- ###
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

if [ "$run" = true ]; then
    echo "Execute CLEO exec with the given bash script"
    for directory_individual in ${rawdirectory}/${subdir_pattern}*; do
        echo "............................................"
        echo "Run CLEO exec"
        echo "in ${directory_individual}"
        {
            ${run_script} ${path2CLEO} ${path2build} ${directory_individual} ${path2exec}
        } || {
            echo "============================================"
            echo "EXCECUTION ERROR: in ${directory_individual}"
            echo "============================================"
        }
        echo "............................................"
    done
    echo "============================================"
fi
### ---------------------------------------------------- ###

echo "--------------------------------------------"
date
echo "END RUN"
