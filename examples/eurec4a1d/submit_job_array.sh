#!/bin/bash

# --------------------------------
# Set the flags to define the steps to run
# --------------------------------
create_init=false   # Create the initial files
run=true            # Run the model

# --------------------------------
# Path to python scripts to create the initial files and run the model in the EURC4A1D case
# --------------------------------
create_script_path="/home/m/m301096/CLEO/examples/eurec4a1d/scripts/create_init_files_job_array.sh"
run_script_path="/home/m/m301096/CLEO/examples/eurec4a1d/scripts/run_CLEO_job_array.sh"

# --------------------------------
# Set microphysics setup
# --------------------------------
# microphysics="null_microphysics"
# microphysics="condensation"
microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"


# --------------------------------
# Set paths
# --------------------------------
path2CLEO="/home/m/m301096/CLEO/"
path2data=${path2CLEO}/data/output_v3.5/
path2build=${path2CLEO}/build_eurec4a1d/
config_directory="/home/m/m301096/repositories/sdm-eurec4a/data/model/input/output_v3.0/"

# --------------------------------
# Get the number of files in the directory
# --------------------------------

echo "microphysics: ${microphysics}"
echo "path2CLEO: ${path2CLEO}"
echo "path2data: ${path2data}"
echo "path2build: ${path2build}"
echo "config_directory: ${config_directory}"


# =================================
# CREATE INIT
if [ "$create_init" = true ]; then
    echo "Create initial files for EUREC4A1D"
    echo "create_script_path: ${create_script_path}"

    files=(${config_directory}/*.yaml)
    number_of_files=${#files[@]}
    # job array ranges from 0 - max_number
    max_number=$(($number_of_files - 1))
    echo "Number of files and slurm array: ${#files[@]}"

    # Update --array=0-max_number
    sed -i "s/#SBATCH --array=.*/#SBATCH --array=0-${max_number}/" "$create_script_path"
    # Update --ntasks-per-node=1
    sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=1/" "$create_script_path"

    JOBID_create=$(\
        sbatch --export=microphysics=${microphysics},path2CLEO=${path2CLEO},path2data=${path2data},config_directory=${config_directory} \
        ${create_script_path}\
        )

    echo "JOBID: ${JOBID_create}"
fi

# =================================
# PURE RUN
if [ "$run" = true ]; then
    echo "Run CLEO for EUREC4A1D with microphysics: ${microphysics}"
    echo "run_script_path: ${run_script_path}"

    directories=($(find ${path2data} -maxdepth 1 -type d -name 'clusters*' -printf '%P\n' | sort))
    # job array ranges from 0 - max_number
    number_of_directories=${#directories[@]}
    max_number=$(($number_of_directories - 1))

    echo "Number of directories and slurm array: ${#directories[@]}"

    # Update --array=0-max_number
    sed -i "s/#SBATCH --array=.*/#SBATCH --array=0-${max_number}/" "$run_script_path"
    # Update --ntasks-per-node=1
    sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=1/" "$run_script_path"

    JOBID_run=$(\
        sbatch  --export=microphysics=${microphysics},path2CLEO=${path2CLEO},path2data=${path2data},path2build=${path2build} \
        ${run_script_path}\
    )
    echo "JOBID: ${JOBID_run}"
fi # =================================
