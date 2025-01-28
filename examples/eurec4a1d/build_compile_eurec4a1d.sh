#!/bin/bash
#SBATCH --job-name=e1d_build_compile_CLEO
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=um1487
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
build=false
compile=false
run=true

# directory parameters
path2CLEO=${HOME}/CLEO/
path2build=${HOME}/CLEO/build_eurec4a1d_openmp/

# activate scripts
source ${HOME}/.bashrc
source ${path2CLEO}/scripts/bash/src/check_inputs.sh


### -------------- run CLEO parameters ------------- ###
run_script_path=${path2CLEO}/examples/eurec4a1d/scripts/run_CLEO_job_array.sh
path2data=${path2CLEO}/data/output_v4.1/
subdir_pattern="cluster*"

microphysics="null_microphysics"
# microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"
# microphysics="coalbure_condensation_cke"

executable_name="eurec4a1d_${microphysics}"
run_excutable="${path2build}/examples/eurec4a1d/stationary_${microphysics}/src/${executable_name}"


export EUREC4A1D_MICROPHYSICS=${microphysics}
export EUREC4A1D_PATH2DATA=${path2data}
export EUREC4A1D_SUBDIR_PATTERN=${subdir_pattern}

check_args_not_empty "${EUREC4A1D_MICROPHYSICS}" "${EUREC4A1D_PATH2DATA}" "${EUREC4A1D_SUBDIR_PATTERN}"


### -------------- build and compile parameters ------------- ###

buildtype="openmp" # as defined by Kokkos configuration; see below
compilername="intel" # as defined by Kokkos configuration; see below
enabledebug=false # as defined by Kokkos configuration; see below
enableyac=false # as defined by YAC flags; see below
yacyaxtroot=/work/bm1183/m300950/yacyaxt
stacksize_limit=204800 # ulimit -s [stacksize_limit] (kB)
ntasks_per_node=128 # number of tasks per node (cpus which shall be used)

build_clean=false
make_clean=true

### ----------------- define executables --------------- ###
compile_executables="eurec4a1d_null_microphysics eurec4a1d_condensation eurec4a1d_collision_condensation eurec4a1d_coalbure_condensation_small eurec4a1d_coalbure_condensation_large eurec4a1d_coalbure_condensation_cke"
# compile_executables="eurec4a1d_null_microphysics eurec4a1d_condensation eurec4a1d_collision_condensation eurec4a1d_coalbure_condensation_small eurec4a1d_coalbure_condensation_large eurec4a1d_coalbure_condensation_cke"


### ---------------------------------------------------- ###


### ----------------- export inputs -------------------- ###
export CLEO_BUILDTYPE=${buildtype}
export CLEO_COMPILERNAME=${compilername}
export CLEO_PATH2CLEO=${path2CLEO}
export CLEO_PATH2BUILD=${path2build}
export CLEO_ENABLEDEBUG=${enabledebug}
export CLEO_ENABLEYAC=${enableyac}
export CLEO_ENABLEYAC=${enableyac}
export CLEO_STACKSIZE_LIMIT=${stacksize_limit}
export CLEO_NTASKS_PER_NODE=${ntasks_per_node}
export CLEO_RUN_EXECUTABLE=${run_excutable}

if [ ${CLEO_ENABLEYAC} == "true" ]
then
  export CLEO_YACYAXTROOT=${yacyaxtroot}
fi


### -------------------- check inputs ------------------- ###
check_args_not_empty "${CLEO_BUILDTYPE}" "${CLEO_COMPILERNAME}" "${CLEO_ENABLEDEBUG}" "${CLEO_PATH2CLEO}" "${CLEO_PATH2BUILD}" "${CLEO_ENABLEYAC}" "${CLEO_STACKSIZE_LIMIT}" "${CLEO_NTASKS_PER_NODE}" "${compile_executables}" "${CLEO_RUN_EXECUTABLE}"
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
echo "CLEO_STACKSIZE_LIMIT = ${CLEO_STACKSIZE_LIMIT}"
echo "CLEO_NTASKS_PER_NODE = ${CLEO_NTASKS_PER_NODE}"
echo "CLEO_RUN_EXECUTABLE = ${CLEO_RUN_EXECUTABLE}"
echo "compile_executables = ${compile_executables}"
echo "### ------------------------------------------- ###"
### ---------------------------------------------------- ###

### --------------------- build CLEO ------------------- ###
if [ "$build_clean" == true ]; then
  rm -r ${CLEO_PATH2BUILD}
  mkdir ${CLEO_PATH2BUILD}
  mkdir ${CLEO_PATH2BUILD}/bin
fi

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
  compilecmd="${CLEO_PATH2CLEO}/scripts/bash/compile_cleo.sh \"${compile_executables}\" ${make_clean}"
  echo ${compilecmd}
  eval ${compilecmd}
fi
### ---------------------------------------------------- ###


### ----------------- run CLEO ------------------- ###
if [ "$run" == true ]; then

  # Get the number of sub directories and create SLURM ARRAY job
  # find all subdirectories
  directories=($(find ${EUREC4A1D_PATH2DATA}/${EUREC4A1D_MICROPHYSICS} -maxdepth 1 -type d -name ${subdir_pattern} -printf '%P\n' | sort))

  # job array ranges from 0 - max_number
  number_of_directories=${#directories[@]}
  max_number=$(($number_of_directories - 1))
  max_number=2

  echo "Number dir:  ${number_of_directories}"
  echo "Slurm Array: --array=0-${max_number}"

  # Update --array=0-max_number
  sed -i "s/#SBATCH --array=.*/#SBATCH --array=0-${max_number}/" "$run_script_path"
  # Update --ntasks-per-node to given value
  sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=${ntasks_per_node}/" "$run_script_path"

  # Submit job array
  JOBID_run=$(sbatch ${run_script_path})
  echo "JOBID CLEO run: ${JOBID_run}"
fi

echo "--------------------------------------------"
date
echo "END RUN"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
