#!/bin/bash
#SBATCH --job-name=e1d_create_init
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=40G
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
microphysics="null_microphysics"
# microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"
# microphysics="coalbure_condensation_cke"

path2CLEO=${HOME}/CLEO/
path2data=${path2CLEO}data/output_v4.1/
path2input=${path2sdmeurec4a}data/model/input_v4.1/


path2output=${path2data}/${microphysics}
path2eurec4a1d=${path2CLEO}examples/eurec4a1d/
path2sdmeurec4a=${HOME}/repositories/sdm-eurec4a/

echo "============================================"
echo "path2CLEO: ${path2CLEO}"
echo "path2data: ${path2data}"
echo "path2eurec4a1d: ${path2eurec4a1d}"
echo "path2sdmeurec4a: ${path2sdmeurec4a}"
echo "microphysics: ${microphysics}"

path2pythonscript=${path2eurec4a1d}scripts/create_model_input_mpi4py.py
echo "path2pythonscript: ${path2pythonscript}"

echo "============================================"
echo "path2output: ${path2output}"
echo "path2input: ${path2input}"

echo "============================================"
default_config_path=${path2eurec4a1d}/default_config/eurec4a1d_config_stationary.yaml

# validate the breakup file path exists
if [ ! -f ${default_config_path} ]; then
    echo "config file path does not exist: ${default_config_path}"
    exit 1
fi

echo "default config path: ${default_config_path}"
if [ "${microphysics}" == "null_microphysics" ] || [ "${microphysics}" == "condensation" ]; then
    breakup_file_path=${path2eurec4a1d}/default_config/breakup.yaml
else
    breakup_file_path=${path2eurec4a1d}/stationary_${microphysics}/src/breakup.yaml
fi

# validate the breakup file path exists
if [ ! -f ${breakup_file_path} ]; then
    echo "breakup file path does not exist: ${breakup_file_path}"
    exit 1
fi

echo "breakup file path: ${breakup_file_path}"


### ---------------------------------------------------- ###
echo "============================================"

### ---- Creation of init files

mpirun -np 40 python ${path2pythonscript} --input_dir_path ${path2input} --output_dir_path ${path2output} --breakup_config_file_path ${breakup_file_path} --default_config_file_path ${default_config_path}
