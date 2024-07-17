#!/bin/bash

# Path to script B
# script_path="/home/m/m301096/CLEO/examples/eurec4a1d/create_init_files_array.sh"
script_path="/home/m/m301096/CLEO/examples/eurec4a1d/run_executable_CLEO_array.sh"

# Set microphysics setup
microphysics="null_microphysics"
# microphysics="condensation"
# microphysics="collision_condensation"
# sed -i "s/microphysics=.*/microphysics=${microphysics}/" "$script_path"

config_directory="/home/m/m301096/repositories/sdm-eurec4a/data/model/input/output_v3.0/"
path2CLEO="/home/m/m301096/CLEO/"
path2data=${path2CLEO}/data/output_v3.5/

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


sbatch --export=microphysics=${microphysics},config_directory=${config_directory},path2CLEO=${path2CLEO},path2data=${path2data} \
    ${script_path}
