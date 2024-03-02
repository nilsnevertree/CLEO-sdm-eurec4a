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

### ---------------------------------------------------- ###
### ------- You MUST edit these lines to set your ------ ###
### --- default compiler(s) (and python environment) --- ###
### ----  and paths for CLEO and build directories  ---- ###
### ---------------------------------------------------- ###
module load gcc/11.2.0-gcc-11.2.0
module load nvhpc/23.9-gcc-11.2.0
spack load cmake@3.23.1%gcc
source activate /work/mh1126/m300950/condaenvs/cleoenv
path2CLEO=${HOME}/CLEO/
path2build=${HOME}/CLEO/roughpaper/build/
gxx="/sw/spack-levante/gcc-11.2.0-bcn7mb/bin/g++"
gcc="/sw/spack-levante/gcc-11.2.0-bcn7mb/bin/gcc"
### ---------------------------------------------------- ###

### ---------------------------------------------------- ###
### ------- You can optionally edit the following ------ ###
### -------- lines to customise your compiler(s) ------- ###
###  ------------ and build configuration  ------------- ###
### ---------------------------------------------------- ###

### --------------- choose CUDA compiler --------------- ###
# set path to Kokkos nvcc wrapper (usually Kokkos bin directory of kokkos after installation)
CLEO_NVCC_WRAPPER="${HOME}/CLEO/roughpaper/build/_deps/kokkos-src/bin/nvcc_wrapper"

# set nvcc compiler used by Kokkos nvcc wrapper as CLEO_CUDA_ROOT/bin/nvcc
# NOTE(!) this path should correspond to the loaded nvhpc module.
# you can get a clue for the correct path e.g. via 'spack find -p nvhpc@23.9'
CLEO_CUDA_ROOT="/sw/spack-levante/nvhpc-23.9-xpxqeo/Linux_x86_64/23.9/cuda/"
### ---------------------------------------------------- ###

### ------------ choose extra compiler flags ----------- ###
CMAKE_CXX_FLAGS="-Werror -Wall -pedantic -g -gdwarf-4 -O0 -mpc64"      # correctness and debugging (note -gdwarf-4 not possible for nvc++)
# CMAKE_CXX_FLAGS="-Werror -Wall -pedantic -O3"                        # performance
### ---------------------------------------------------- ###

### ------------ choose Kokkos configuration ----------- ###
# flags for serial kokkos
kokkosflags="-DKokkos_ARCH_NATIVE=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_SERIAL=ON"

# flags for host parallelism (e.g. using OpenMP)
kokkoshost="-DKokkos_ENABLE_OPENMP=ON"

# flags for device parallelism (e.g. on gpus)
kokkosdevice="-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=O -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
-DCLEO_CUDA_ROOT=${CLEO_CUDA_ROOT} -DCLEO_NVCC_WRAPPER=${CLEO_NVCC_WRAPPER}"
### ---------------------------------------------------- ###

### ------------ build and compile with cmake ---------- ###
echo "CLEO_CXX_COMPILER=${gxx} CLEO_CC_COMPILER=${gcc}"
echo "CUDA=${CLEO_CUDA_ROOT}/bin/nvcc (via Kokkos nvcc wrapper)"
echo "CLEO_NVCC_WRAPPER=${CLEO_NVCC_WRAPPER}"
echo "BUILD_DIR: ${path2build}"
echo "KOKKOS_FLAGS: ${kokkosflags}"
echo "KOKKOS_DEVICE_PARALLELISM: ${kokkosdevice}"
echo "KOKKOS_HOST_PARALLELISM: ${kokkoshost}"
echo "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}"

# build then compile in parallel
cmake -DCMAKE_CXX_COMPILER=${gxx} \
    -DCMAKE_CC_COMPILER=${gcc} \
    -DCLEO_CXX_COMPILER=${gxx} \
    -DCLEO_CC_COMPILER=${gcc} \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
    -S ${path2CLEO} -B ${path2build} \
    ${kokkosflags} ${kokkosdevice} ${kokkoshost} && \
    cmake --build ${path2build} --parallel
### ---------------------------------------------------- ###
