#!/bin/bash
#SBATCH --job-name=eurec4a1d
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eurec4a1d.%j_out.out
#SBATCH --error=./logfiles/eurec4a1d.%j_err.out

<<<<<<< HEAD
### ---------------------------------------------------- ###
=======
# TODO(all): python script(s) for example

>>>>>>> 6e092c6475db4364703c05d82843cc96d2e10f3f
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
path2CLEO=${HOME}/CLEO/
path2build=${HOME}/CLEO/build_eurec4a1D/
enableyac=false

executables="eurec4a1D"

path2data=${path2CLEO}data/newoutput/
path2eurec4a1d=${path2CLEO}examples/eurec4a1d/

# cloud type
path2sdmeurec4a=${HOME}/repositories/sdm-eurec4a/
cloud_config_directory=${path2sdmeurec4a}data/model/input/new_subset/
# cloud_observation_configfile=${path2sdmeurec4a}data/model/input/new/clusters_18.yaml
# cloud_observation_configfile=${cloud_config_directory}clusters_18.yaml

# Use the stationary or evolving version of the model

### ---------- Setup for the EUREC4A1D model ---------- ###

# --- stationary version, with super droplet creation at domain top by boundarz conditions
path2build=${path2builds}build_eurec4a1D_stationary_condensation/
rawdirectory=${path2data}stationary_condensation/

pythonscript=${path2eurec4a1d}scripts/eurec4a1d_stationary.py
configfile=${path2eurec4a1d}src/config/eurec4a1d_config_stationary.yaml

# # --- evolving version, without super droplet creation at domain top
# path2build=${path2builds}build_eurec4a1D_evolving/
# pythonscript=${path2eurec4a1d}scripts/eurec4a1d_evolving.py
# configfile=${path2eurec4a1d}src/config/eurec4a1d_config_evolving.yaml
# rawdirectory=${path2data}evolving/

# create the script arguments
# script_args="${HOME} ${configfile} ${cloud_observation_configfile} ${rawdirectory}"
### ---------------------------------------------------- ###


### ------------------ Load Modules -------------------- ###
cleoenv=/work/mh1126/m300950/cleoenv
python=${cleoenv}/bin/python3
spack load cmake@3.23.1%gcc
module load python3/2022.01-gcc-11.2.0
source activate ${cleoenv}

### ---------- build, compile and run example ---------- ###
${path2CLEO}/examples/run_example.sh \
  ${buildtype} ${path2CLEO} ${path2build} ${enableyac} \
  "${executables}" ${pythonscript} "${script_args}"
### ---------------------------------------------------- ###

### -------------------- print inputs ------------------ ###
echo "----- Running Example -----"
echo "buildtype:  ${buildtype}"
echo "path2CLEO: ${path2CLEO}"
echo "path2build: ${path2build}"
echo "executables: ${executables}"
echo "pythonscript: ${pythonscript}"
# echo "script_args: ${script_args}"
echo "---------------------------"
### ---------------------------------------------------- ###

# ## ---------------------- build CLEO ------------------ ###
# ${path2CLEO}/scripts/bash/build_cleo.sh ${buildtype} ${path2CLEO} ${path2build}
# ### ---------------------------------------------------- ###

# ### --------- compile executable(s) from scratch ---------- ###
# cd ${path2build} && make clean

# ${path2CLEO}/scripts/bash/compile_cleo.sh ${cleoenv} ${buildtype} ${path2build} "${executables}"
# # ### ---------------------------------------------------- ###

### --------- run model through Python script ---------- ###
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

for cloud_configfile in ${cloud_config_directory}/*.yaml; do
    echo "Running rainshaft1d.py with ${cloud_configfile}"
    script_args="${HOME} ${configfile} ${cloud_configfile} ${rawdirectory}"

    {
        ${python}  ${pythonscript} ${path2CLEO} ${path2build} ${script_args}

    } || {
        echo "Error in running rainshaft1d.py with ${cloud_configfile}"
    }
done

### ---------------------------------------------------- ###

echo "--------------------------------------------"
echo "END RUN"
date
echo "============================================"
