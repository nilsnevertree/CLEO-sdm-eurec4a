#!/bin/bash
#SBATCH --job-name=gpurunCLEO
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --mail-user=clara.bayley@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./gpurunCLEO_out.%j.out
#SBATCH --error=./gpurunCLEO_err.%j.out

### ------------- PLEASE NOTE: this script assumes you ------------- ###
### ------------- have already built CLEO in path2build ------------ ###
### -------------------  directory using cmake  -------------------- ###

### ----- You need to edit these lines to set your ----- ###
### ----- default compiler and python environment   ---- ###
### ----  and paths for CLEO and build directories  ---- ###
module load gcc/11.2.0-gcc-11.2.0
module load python3/2022.01-gcc-11.2.0
module load nvhpc/23.7-gcc-11.2.0
spack load cmake@3.23.1%gcc
source activate /work/mh1126/m300950/condaenvs/cleoenv

path2CLEO=${HOME}/CLEO/
path2build=$1 # get from command line args
configfile=${HOME}/CLEO/src/config/config.txt

python=python
### ---------------------------------------------------- ###

### ------------------- compile_run.sh ----------------- ###
if [ "${path2build}" == "" ]
then
  path2build=${HOME}/CLEO/build/
fi

### compile CLEO in ./build directory
echo "path to build directory: ${path2build}"
cd ${path2build} && pwd
make -j 128

### run CLEO
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
runcmd="${path2build}/src/runCLEO ${configfile}"
echo ${runcmd}
${runcmd}
### ---------------------------------------------------- ###
