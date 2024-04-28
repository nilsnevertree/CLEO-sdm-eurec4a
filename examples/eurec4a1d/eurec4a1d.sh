#!/bin/bash
#SBATCH --job-name=eurec4a1d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=128
#SBATCH --mem=30G
#SBATCH --time=00:05:00
#SBATCH --mail-user=clara.bayley@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./eurec4a1d_out.%j.out
#SBATCH --error=./eurec4a1d_err.%j.out

### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ---- build type, directories, the executable(s) ---- ###
### -------- to compile, and your python script -------- ###
### ---------------------------------------------------- ###
buildtype="cuda"
path2CLEO=${HOME}/CLEO/
path2build=${HOME}/CLEO/build_eurec4a1D/
executables="eurec4a1D"

configfile=${path2CLEO}/examples/eurec4a1d/src/config/eurec4a1d_config.yaml
pythonscript=""
script_args=""
### ---------------------------------------------------- ###
### ---------------------------------------------------- ###
### ---------------------------------------------------- ###

### ---------- build, compile and run example ---------- ###
${path2CLEO}/examples/run_example.sh \
  ${buildtype} ${path2CLEO} ${path2build} \
  "${executables}" ${pythonscript} "${script_args}"
### ---------------------------------------------------- ###
