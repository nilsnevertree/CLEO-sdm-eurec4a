#!/bin/bash
#SBATCH --job-name=roughCLEO
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=30G
#SBATCH --time=00:05:00
#SBATCH --mail-user=clara.bayley@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./gpubuildCLEO_out.%j.out
#SBATCH --error=./gpubuildCLEO_err.%j.out

### ----- You need to edit these lines to set your ----- ###
### ----- default compiler and python environment   ---- ###
### ----  and paths for CLEO and build directories  ---- ###
module load gcc/11.2.0-gcc-11.2.0
module load nvhpc/23.9-gcc-11.2.0
spack load cmake@3.23.1%gcc
source activate /work/mh1126/m300950/condaenvs/cleoenv
path2build=${HOME}/CLEO/roughpaper/build/
gxx="g++"
gcc="gcc"
### ---------------------------------------------------- ###

### ------------ choose CUDA compiler ----------- ###
# set nvcc compiler used by Kokkos nvcc wrapper as CUDA_ROOT/bin/nvcc
# NOTE(!) this path should correspond to the loaded nvhpc module. Get path e.g. via 'spack find -p nvhpc@23.9'
CUDA_ROOT="/sw/spack-levante/nvhpc-23.9-xpxqeo/Linux_x86_64/23.9/cuda/"
### ---------------------------------------------------- ###

### ------------ choose Kokkos configuration ----------- ###
kokkosflags="-DKokkos_ARCH_NATIVE=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_SERIAL=ON"                 # serial kokkos
kokkosdevice="-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=ON -DCUDA_ROOT=${CUDA_ROOT}" # flags for device parallelism (e.g. on gpus)
# kokkoshost="-DKokkos_ENABLE_OPENMP=ON"                                                                  # flags for host parallelism (e.g. using OpenMP)
### ---------------------------------------------------- ###

### ------------ choose extra compiler flags ----------- ###
flags="-g -O0 -mpc64"                                            # correctness
# flags="-O3"                                                    # performance
### ---------------------------------------------------- ###

### ------------------ build_compile.sh ---------------- ###
### build CLEO using cmake (with optional thread parallelism through Kokkos)
echo "CXX=${gxx} CC=${gcc}"
echo "CUDA=${CUDA_ROOT}/bin/nvcc (via Kokkos nvcc wrapper)"
echo "BUILD_DIR: ${path2build}"
echo "KOKKOS_FLAGS: ${kokkosflags}"
echo "KOKKOS_DEVICE_PARALLELISM: ${kokkosdevice}}"
echo "KOKKOS_HOST_PARALLELISM: ${kokkoshost}"
echo "CXX_COMPILER_FLAGS: ${flags}"

cmake -DCMAKE_CXX_COMPILER=${gxx} \
    -DCMAKE_CC_COMPILER=${gcc} \
    -DCMAKE_CXX_FLAGS="${flags}" \
    -S ../ -B ${path2build} \
    ${kokkosflags} ${kokkosdevice} ${kokkoshost} && \
    cmake --build ${path2build}  --parallel

### compile CLEO
cd ${path2build} && pwd
make -j 16
### ---------------------------------------------------- ###
