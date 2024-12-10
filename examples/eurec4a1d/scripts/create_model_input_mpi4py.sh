#!/bin/bash
#SBATCH --job-name=e1d_creat_init
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=../logfiles/create_init_files/mpi4py/.%j_out.out
#SBATCH --error=../logfiles/create_init_files/mpi4py/.%j_err.out

### --------------------- Version --------------------- ###
echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "date: $(date)"
echo "============================================"
### ---------------------------------------------------- ###

source ${HOME}/.bashrc

### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312/
# module purge
conda activate ${env}

pythonpath=${env}/bin/python
echo "Using Python from: $(which python)"
### ---------------------------------------------------- ###

### ------------------ Input Parameters ---------------- ###
microphysics=collision_condensation
path2CLEO=${HOME}/CLEO/
path2data=${path2CLEO}data/debug_output/short_duration_1024/
path2eurec4a1d=${path2CLEO}examples/eurec4a1d/
path2sdmeurec4a=${HOME}/repositories/sdm-eurec4a/
echo "path2CLEO: ${path2CLEO}"
echo "path2data: ${path2data}"
echo "path2eurec4a1d: ${path2eurec4a1d}"
echo "path2sdmeurec4a: ${path2sdmeurec4a}"
echo "microphysics: ${microphysics}"

path2pythonscript=${path2eurec4a1d}scripts/create_model_input_mpi4py.py
echo "path2pythonscript: ${path2pythonscript}"

path2output=${path2data}/${microphysics}
path2input=${path2sdmeurec4a}data/model/input_v4.0/
echo "path2output: ${path2output}"
echo "path2input: ${path2input}"
### ---------------------------------------------------- ###
echo "============================================"

### ---- Creation of init files

mpirun -np 40 python ${path2pythonscript} --input_dir_path ${path2input} --output_dir_path ${path2output}
