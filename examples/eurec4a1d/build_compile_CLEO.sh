#!/bin/bash
#SBATCH --job-name=e1d_build_compile_CLEO
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=1G
#SBATCH --time=00:5:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/build_compile_CLEO/%j/%j_out.out
#SBATCH --error=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/build_compile_CLEO/%j/%j_err.out

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "START RUN"
date
echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "============================================"

set -e
module purge
spack unload --all


# setps to run
build=true
compile=true


# run parameters
buildtype="openmp" # as defined by Kokkos configuration; see below
compilername="intel" # as defined by Kokkos configuration; see below
path2CLEO=${HOME}/CLEO/
path2build=${HOME}/CLEO/build_eurec4a1d_openmpi/

rm -r ${path2build}
mkdir ${path2build}
mkdir ${path2build}/bin

executables="eurec4a1d_null_microphysics eurec4a1d_condensation eurec4a1d_collision_condensation eurec4a1d_coalbure_condensation_small eurec4a1d_coalbure_condensation_large eurec4a1d_coalbure_condensation_cke"
# executables="eurec4a1d_condensation"
enabledebug=false # as defined by Kokkos configuration; see below
enableyac=false # as defined by YAC flags; see below
yacyaxtroot=/work/bm1183/m300950/yacyaxt
make_clean=true

### ---------------------------------------------------- ###

### ------------------ check arguments ----------------- ###
if [ "${path2CLEO}" == "" ]
then
  echo "Please provide path to CLEO source directory"
  exit 1
fi
source ${path2CLEO}/scripts/bash/src/check_inputs.sh
check_args_not_empty "${buildtype}" "${compilername}" "${enabledebug}" "${path2CLEO}" "${path2build}" "${enableyac}"
### ---------------------------------------------------- ###

### ----------------- export inputs -------------------- ###
export CLEO_BUILDTYPE=${buildtype}
export CLEO_COMPILERNAME=${compilername}
export CLEO_PATH2CLEO=${path2CLEO}
export CLEO_PATH2BUILD=${path2build}
export CLEO_ENABLEDEBUG=${enabledebug}
export CLEO_ENABLEYAC=${enableyac}

if [ ${CLEO_ENABLEYAC} == "true" ]
then
  export CLEO_YACYAXTROOT=${yacyaxtroot}
fi
### ---------------------------------------------------- ###

### -------------------- check inputs ------------------ ###
check_source_and_build_paths
check_buildtype
check_compilername
check_yac
### ---------------------------------------------------- ###


### -------------------- print inputs ------------------- ###
echo "### --------------- User Inputs -------------- ###"
echo "CLEO_BUILDTYPE = ${CLEO_BUILDTYPE}"
echo "CLEO_COMPILERNAME = ${CLEO_COMPILERNAME}"
echo "CLEO_PATH2CLEO = ${CLEO_PATH2CLEO}"
echo "CLEO_PATH2BUILD = ${CLEO_PATH2BUILD}"
echo "CLEO_ENABLEDEBUG = ${CLEO_ENABLEDEBUG}"
echo "CLEO_ENABLEYAC = ${CLEO_ENABLEYAC}"
echo "CLEO_YACYAXTROOT = ${CLEO_YACYAXTROOT}"
echo "executables = ${executables}"
echo "### ------------------------------------------- ###"
### ---------------------------------------------------- ###

### --------------------- build CLEO ------------------- ###
if [ "$build" == true ]; then
    echo "Build CLEO"
    buildcmd="${CLEO_PATH2CLEO}/scripts/bash/build_cleo.sh"
    echo ${buildcmd}
    eval ${buildcmd}
fi
### ---------------------------------------------------- ###

### ---------------- compile executables --------------- ###
if [ "$compile" == true ]; then
    echo "Compile CLEO"
    compilecmd="${CLEO_PATH2CLEO}/scripts/bash/compile_cleo.sh \"${executables}\" ${make_clean}"
    echo ${compilecmd}
    eval ${compilecmd}
fi
### ---------------------------------------------------- ###

echo "--------------------------------------------"
date
echo "END RUN"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
