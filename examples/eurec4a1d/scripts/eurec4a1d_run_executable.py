import os
import sys
from pathlib import Path
import yaml
import subprocess

print(f"Enviroment: {sys.prefix}")

path2CLEO = Path(sys.argv[1])
path2build = Path(sys.argv[2])
cloud_observation_filepath = Path(sys.argv[3])
rawdirectory = Path(sys.argv[4])

with open(cloud_observation_filepath, "r") as f:
    cloud_observation_config = yaml.safe_load(f)


identification_type = cloud_observation_config["cloud"]["identification_type"]
cloud_id = cloud_observation_config["cloud"]["cloud_id"]

rawdirectory_individual = rawdirectory / f"{identification_type}_{cloud_id}"
config_dir = rawdirectory_individual / "config"

config_file = config_dir / "eurec4a1d_config.yaml"
dataset_file_str = str(rawdirectory_individual / "eurec4a1d_sol.zarr")

with open(config_file, "r") as f:
    config_dict = yaml.safe_load(f)


print("===============================")
print("Running CLEO executable")
os.chdir(path2build)
os.system("pwd")
os.system("rm -rf " + dataset_file_str)  # delete any existing dataset
executable = str(path2build) + "/examples/eurec4a1d/src/eurec4a1D"
print("Executable: " + executable)
print("Config file: " + str(config_file))
os.system(executable + " " + str(config_file))

proc = subprocess.Popen(
    args=[executable + " " + str(config_file)],
    stdout=subprocess.PIPE,
)

(out, err) = proc.communicate()

print("===============================")
