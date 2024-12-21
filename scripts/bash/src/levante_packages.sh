#!/bin/bash

### -------------- GCC compiler(s) Packages ------------ ###
levante_gcc=gcc/11.2.0-gcc-11.2.0 # bcn7mbu # module load []
levante_gcc_cmake=cmake@3.26.3%gcc@=11.2.0/fuvwuhz # spack load []
levante_gcc_openmpi=openmpi@4.1.2%gcc@11.2.0 # spack load []
levante_gxx_compiler="/sw/spack-levante/openmpi-4.1.2-mnmady/bin/mpic++"
levante_gcc_compiler="/sw/spack-levante/openmpi-4.1.2-mnmady/bin/mpicc"
levante_gcc_cuda=cuda@12.2.0%gcc@=11.2.0  # spack load []
levante_gcc_cuda_root="/sw/spack-levante/cuda-12.2.0-2ttufp/" # [cuda_root]/bin/nvcc") (can get hint for correct path via 'spack find -p nvhpc@23.9')
levante_gcc_netcdf_yac=netcdf-c/4.8.1-openmpi-4.1.2-gcc-11.2.0 # module load []
levante_gcc_openblas_yac=openblas@0.3.18%gcc@=11.2.0 # spack load []
### ---------------------------------------------------- ###

### ------------- Intel compiler(s) Packages ------------ ###
levante_intel=intel-oneapi-compilers/2023.2.1-gcc-11.2.0 # module load []
levante_intel_cmake=cmake@3.23.1%oneapi # spack load []
levante_intel_openmpi=openmpi@4.1.5%oneapi # spack load []
levante_icpc_compiler="/sw/spack-levante/openmpi-4.1.6-ux3zoj/bin/mpic++"
levante_icc_compiler="/sw/spack-levante/openmpi-4.1.6-ux3zoj/bin/mpicc"
### ---------------------------------------------------- ###
