#!/bin/bash

# Path to script B
script_path="/home/m/m301096/CLEO/examples/eurec4a1d/scripts/create_init_files_job_array.sh"
# script_path="/home/m/m301096/CLEO/examples/eurec4a1d/scripts/run_CLEO_job_array.sh"

# Set microphysics setup
# microphysics="null_microphysics"
microphysics="condensation"
# microphysics="collision_condensation"

path2CLEO="/home/m/m301096/CLEO/"
path2data=${path2CLEO}/data/output_v3.5/
path2build=${path2CLEO}/build_eurec4a1d/
config_directory="/home/m/m301096/repositories/sdm-eurec4a/data/model/input/output_v3.0/"

# Get the number of files in the directory

files=(${config_directory}/*.yaml)
# Subtract 1 from the number of files to get the maximum number for the array
number_of_files=${#files[@]}
max_number=$(($number_of_files - 1))
# max_number=1

# Update --array=0-max_number
sed -i "s/#SBATCH --array=.*/#SBATCH --array=0-${max_number}/" "$script_path"

# Update --ntasks-per-node=1
sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=1/" "$script_path"


sbatch --export=microphysics=${microphysics},path2CLEO=${path2CLEO},path2data=${path2data},path2build=${path2build},config_directory=${config_directory} \
    ${script_path}
