#!/bin/bash
#SBATCH --job-name=rain1d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=128
#SBATCH --mem=30G
#SBATCH --time=00:10:00
#SBATCH --mail-user=clara.bayley@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./rain1d_out.%j.out
#SBATCH --error=./rain1d_err.%j.out

### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ---- build type, directories, the executable(s) ---- ###
### -------- to compile, and your python script -------- ###
### ---------------------------------------------------- ###
buildtype="cuda"
path2CLEO=${HOME}/CLEO/
path2build=${HOME}/CLEO/build_rshaft1D/
executables="rshaft1D"

pythonscript=${path2CLEO}/examples/rainshaft1d/rainshaft1d.py
configfile=${path2CLEO}/examples/rainshaft1d/src/config/rain1d_config.yaml
script_args="${configfile}"
### ---------------------------------------------------- ###
### ---------------------------------------------------- ###
### ---------------------------------------------------- ###

### ---------- build, compile and run example ---------- ###
${path2CLEO}/examples/run_example.sh \
  ${buildtype} ${path2CLEO} ${path2build} \
  "${executables}" ${pythonscript} "${script_args}"
### ---------------------------------------------------- ###

echo "--------------------------------------------"
echo "END RUN"
date
echo "============================================"
