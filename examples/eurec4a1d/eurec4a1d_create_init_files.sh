#!/bin/bash
#SBATCH --job-name=eurec4a1d_create_init_files
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eurec4a1d_create_init_files.%j_out.out
#SBATCH --error=./logfiles/eurec4a1d_create_init_files.%j_err.out


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

path2CLEO=${HOME}/CLEO/
path2builds=${path2CLEO}builds/
path2data=${path2CLEO}data/output_v3.4/
path2eurec4a1d=${path2CLEO}examples/eurec4a1d/

# creation script
pythonscript=${path2eurec4a1d}scripts/eurec4a1d_stationary.py

# baseline config file
configfile=${path2eurec4a1d}src/config/eurec4a1d_config_stationary.yaml

# ----- Directory for cloud configuration files ------ #
path2sdmeurec4a=${HOME}/repositories/sdm-eurec4a/
cloud_config_directory=${path2sdmeurec4a}data/model/input/output_v3.0/
# ---------------------------------------------------- #


### ---------- Setup for the EUREC4A1D model ---------- ###

# --- stationary version, with super droplet creation at domain top by boundarz conditions

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


### ------------------ Create input -------------------- ###
for cloud_configfile in ${cloud_config_directory}/*.yaml; do
    echo "::::::::::::::::::::::::::::::::::::::::::::"
    echo "New config files."
    echo "Prepare eurec4a1d config files with: ${cloud_configfile}"

    script_args="${HOME} ${configfile} ${cloud_configfile} ${rawdirectory}"
    {
        ${python}  ${pythonscript} ${path2CLEO} ${path2build} ${script_args}

    } || {
        echo "============================================"
        echo "ERROR: in ${cloud_configfile}"
        echo "============================================"
    }
    echo "::::::::::::::::::::::::::::::::::::::::::::"
done
### ---------------------------------------------------- ###

echo "--------------------------------------------"
date
echo "END RUN"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
