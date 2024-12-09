#!/bin/bash
#SBATCH --job-name=eurec4a1d_update_config
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eurec4a1d_update_config.%j_out.out
#SBATCH --error=./logfiles/eurec4a1d_update_config.%j_err.out

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

# set paths
path2CLEO=${HOME}/CLEO/
path2data=${path2CLEO}data/output_v3.2/
path2eurec4a1d=${path2CLEO}examples/eurec4a1d/
subdir_pattern=clusters_

# python script to run
pythonscript=${path2eurec4a1d}scripts/eurec4a1d_update_config.py

### ---------- Setup for the EUREC4A1D model ---------- ###
# Use the stationary setup of the model

# # NO PHYSICS
# rawdirectory=${path2data}stationary_no_physics/

# CONDENSTATION
# rawdirectory=${path2data}stationary_condensation/

# # COLLISION AND CONDENSTATION
rawdirectory=${path2data}stationary_collision_condensation/

### ---------------------------------------------------- ###



### ------------------ Load Modules -------------------- ###
cleoenv=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
python=${cleoenv}/bin/python3
source activate ${cleoenv}
### ---------------------------------------------------- ###

### -------------------- print inputs ------------------ ###
echo "----- Update Config Files -----"
echo "path2CLEO: ${path2CLEO}"
echo "pythonscript: ${pythonscript}"
echo "---------------------------"
### --------------------------------------------------- ###


echo "Update Config Files"
for exp_folder in ${rawdirectory}/${subdir_pattern}*; do
    echo "::::::::::::::::::::::::::::::::::::::::::::"
    echo "UPDATE CONFIG FILE"
    echo "in ${exp_folder}"
    {
        ${python}  ${pythonscript} ${exp_folder}
    } || {
        echo "============================================"
        echo "EXCECUTION ERROR: in ${exp_folder}"
        echo "============================================"
    }
    echo "::::::::::::::::::::::::::::::::::::::::::::"
done
### ---------------------------------------------------- ###

echo "--------------------------------------------"
date
echo "END RUN"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
