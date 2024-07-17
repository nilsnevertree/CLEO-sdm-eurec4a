#!/bin/bash

# Path to script B
create_script_path="/home/m/m301096/CLEO/examples/eurec4a1d/scripts/create_init_files_job_array.sh"
run_script_path="/home/m/m301096/CLEO/examples/eurec4a1d/scripts/run_CLEO_job_array.sh"

# Set microphysics setup
# microphysics="null_microphysics"
# microphysics="condensation"
# microphysics="collision_condensation"
microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"


path2CLEO="/home/m/m301096/CLEO/"
path2data=${path2CLEO}/data/output_v3.5/
path2build=${path2CLEO}/build_eurec4a1d_new/
config_directory="/home/m/m301096/repositories/sdm-eurec4a/data/model/input/output_v3.0/"

# Get the number of files in the directory

files=(${config_directory}/*.yaml)
# Subtract 1 from the number of files to get the maximum number for the array
number_of_files=${#files[@]}
max_number=$(($number_of_files - 1))
# max_number=1

# =================================
# CREATE INIT

# Update --array=0-max_number
sed -i "s/#SBATCH --array=.*/#SBATCH --array=0-${max_number}/" "$create_script_path"
# Update --ntasks-per-node=1
sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=1/" "$create_script_path"

echo "CREATE INIT SCRIPT"
JOBID_create=$(\
    sbatch --export=microphysics=${microphysics},path2CLEO=${path2CLEO},path2data=${path2data},config_directory=${config_directory} \
    ${create_script_path}\
    )

echo "JOBID: ${JOBID_create}"

# =================================
# PURE RUN

# # Update --array=0-max_number
# sed -i "s/#SBATCH --array=.*/#SBATCH --array=0-${max_number}/" "$run_script_path"
# # Update --ntasks-per-node=1
# sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=1/" "$run_script_path"

# echo "RUN SCRIPT"
# JOBID_run=$(\
#     sbatch  --export=microphysics=${microphysics},path2CLEO=${path2CLEO},path2data=${path2data},path2build=${path2build} \
#     ${run_script_path}\
# )
# echo "JOBID: ${JOBID_run}"

# =================================
