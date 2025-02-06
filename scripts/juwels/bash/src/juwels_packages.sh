#!/bin/bash

### -------------- GCC compiler(s) Packages ------------ ###
juwels_gcc=GCC/13.3.0 # module load
juwels_gcc_cmake=CMake/3.29.3 # module load
juwels_gcc_openmpi=OpenMPI/5.0.5 # module load
juwels_gxx_compiler="/p/software/default/stages/2025/software/OpenMPI/5.0.5-GCC-13.3.0/bin/mpic++"
juwels_gcc_compiler="/p/software/default/stages/2025/software/OpenMPI/5.0.5-GCC-13.3.0/bin/mpicc"
### ---------------------------------------------------- ###

### ------------- Intel compiler(s) Packages ------------ ###
levante_intel=intel-oneapi-compilers/2023.2.1-gcc-11.2.0 # module load
levante_intel_cmake=cmake@3.23.1%oneapi # spack load
levante_intel_openmpi=openmpi@4.1.5%oneapi # spack load
levante_icpc_compiler="/sw/spack-levante/openmpi-4.1.6-ux3zoj/bin/mpic++"
levante_icc_compiler="/sw/spack-levante/openmpi-4.1.6-ux3zoj/bin/mpicc"
### ---------------------------------------------------- ###
