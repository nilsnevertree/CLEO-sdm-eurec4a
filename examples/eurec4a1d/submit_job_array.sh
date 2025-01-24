#!/bin/bash
#SBATCH --job-name=e1d_submit_master
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --mem=10M
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=um1487
#SBATCH --output=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/submit_master/%j_out.out
#SBATCH --error=/home/m/m301096/CLEO/examples/eurec4a1d/logfiles/submit_master/%j_err.out

# --------------------------------
# Set the flags to define the steps to run
# --------------------------------
run=true          # Run the model

path2CLEO=${HOME}/CLEO

# --------------------------------
# Path to python scripts to create the initial files and run the model in the EURC4A1D case
# --------------------------------
run_script_path=${path2CLEO}/examples/eurec4a1d/scripts/run_CLEO_job_array.sh
CLEO_ENABLEYAC=false
stacksize_limit=204800 # ulimit -s [stacksize_limit] (kB)
# --------------------------------
# Set microphysics setup
# --------------------------------
microphysics="null_microphysics"
# microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"
# microphysics="coalbure_condensation_cke"


# --------------------------------
# Set paths
# --------------------------------
path2CLEO="/home/m/m301096/CLEO/"
path2data=${path2CLEO}/data/output_v4.1/
path2microphysics=${path2data}/${microphysics}
path2build=${path2CLEO}/build_eurec4a1d_openmpi/
# config_directory="/home/m/m301096/repositories/sdm-eurec4a/data/model/input/output_v3.0/"

# --------------------------------
# Get the number of files in the directory
# --------------------------------

echo "--------------------------------"
echo "microphysics: ${microphysics}"
echo "path2CLEO: ${path2CLEO}"
echo "path2data: ${path2data}"
echo "path2microphysics: ${path2microphysics}"
echo "path2build: ${path2build}"
# echo "config_directory: ${config_directory}"
echo "--------------------------------"

# =================================

echo "Run CLEO for EUREC4A1D with microphysics: ${microphysics}"
echo "run_script_path: ${run_script_path}"

directories=($(find ${path2microphysics} -maxdepth 1 -type d -name 'cluster*' -printf '%P\n' | sort))
# echo "Directories: ${directories[@]}"
# job array ranges from 0 - max_number
number_of_directories=${#directories[@]}
# number_of_directories=8
max_number=$(($number_of_directories - 1))

echo "Number of directories and slurm array: ${number_of_directories}"

# Update --array=0-max_number
sed -i "s/#SBATCH --array=.*/#SBATCH --array=0-${max_number}/" "$run_script_path"
# Update --ntasks-per-node=1
sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=1/" "$run_script_path"

JOBID_run=$(\
    sbatch  --export=microphysics=${microphysics},path2CLEO=${path2CLEO},path2data=${path2data},path2build=${path2build},stacksize_limit=${stacksize_limit},CLEO_ENABLEYAC=${CLEO_ENABLEYAC} \
    ${run_script_path}\
)
echo "JOBID: ${JOBID_run}"
