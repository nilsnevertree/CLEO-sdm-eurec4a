'''
----- CLEO -----
File: yac1_fromfile.py
Project: fromfile
Created Date: Friday 17th November 2023
Author: Clara Bayley (CB)
Additional Contributors:
-----
Last Modified: Tuesday 9th April 2024
Modified By: CB
-----
License: BSD 3-Clause "New" or "Revised" License
https://opensource.org/licenses/BSD-3-Clause
-----
Copyright (c) 2023 MPI-M, Clara Bayley
-----
File Description:
Script runs CLEO executable for 3D example with time varying thermodynamics
read from binary files to test that YAC can send the data to CLEO correctly.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path2CLEO = sys.argv[1]
path2build = sys.argv[2]
configfile = sys.argv[3]

sys.path.append(path2CLEO)  # for imports from pySD package
# for imports from example plotting package
sys.path.append(path2CLEO+"/examples/exampleplotting/")

from src import gen_input_thermo, plot_output_thermo
from plotssrc import pltsds, pltmoms
from pySD.sdmout_src import *
from pySD.gbxboundariesbinary_src import read_gbxboundaries as rgrid
from pySD.gbxboundariesbinary_src import create_gbxboundaries as cgrid
from pySD.initsuperdropsbinary_src import *
from pySD.initsuperdropsbinary_src import create_initsuperdrops as csupers
from pySD.thermobinary_src import create_thermodynamics as cthermo
from pySD.thermobinary_src import read_thermodynamics as rthermo

### ---------------------------------------------------------------- ###
### ----------------------- INPUT PARAMETERS ----------------------- ###
### ---------------------------------------------------------------- ###
### --- essential paths and filenames --- ###
# path and filenames for creating initial SD conditions
constsfile = path2CLEO+"/libs/cleoconstants.hpp"
binpath = path2build+"/bin/"
sharepath = path2build+"/share/"
gridfile = sharepath+"yac1_dimlessGBxboundaries.dat"
initSDsfile = sharepath+"yac1_dimlessSDsinit.dat"
thermofile = sharepath+"/yac1_dimlessthermo.dat"

# path and file names for plotting results
setupfile = binpath+"yac1_setup.txt"
dataset = binpath+"yac1_sol.zarr"

### --- plotting initialisation figures --- ###
# booleans for [making, saving] initialisation figures
isfigures = [True, True]
savefigpath = path2build+"/bin/"  # directory for saving figures

### --- settings for 2-D gridbox boundaries --- ###
zgrid = [0, 1500, 60]           # evenly spaced zhalf coords [zmin, zmax, zdelta] [m]
xgrid = [0, 1500, 50]           # evenly spaced xhalf coords [m]
ygrid = np.array([0, 100, 200, 300])   # array of yhalf coords [m]

### --- settings for initial superdroplets --- ###
# settings for initial superdroplet coordinates
zlim = 1000             # max z coord of superdroplets
npergbx = 2             # number of superdroplets per gridbox

monor = 1e-6            # all SDs have this same radius [m]
dryr_sf = 1.0           # scale factor for dry radii [m]
numconc = 5e8           # total no. conc of real droplets [m^-3]
randcoord = False       # sample SD spatial coordinates randomly or not

### --- settings for 2D Thermodynamics --- ###
PRESSz0 = 101500 # [Pa]
TEMPz0 = 300     # [K]
qvapz0 = 0.05    # [Kg/Kg]
qcondz0 = 0.001  # [Kg/Kg]
WMAX = 1.5       # [m/s]
Zlength = 1500   # [m]
Xlength = 1500   # [m]
VMAX = 1.0       # [m/s]
Ylength = 300    # [m]
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###

### ---------------------------------------------------------------- ###
### ------------------- BINARY FILES GENERATION--------------------- ###
### ---------------------------------------------------------------- ###
### --- ensure build, share and bin directories exist --- ###
if path2CLEO == path2build:
    raise ValueError("build directory cannot be CLEO")
else:
    Path(path2build).mkdir(exist_ok=True)
    Path(sharepath).mkdir(exist_ok=True)
    Path(binpath).mkdir(exist_ok=True)
    if isfigures[1]:
        Path(savefigpath).mkdir(exist_ok=True)

### --- delete any existing initial conditions --- ###
os.system("rm "+gridfile)
os.system("rm "+initSDsfile)
os.system("rm "+thermofile[:-4]+"*")

### ----- write gridbox boundaries binary ----- ###
cgrid.write_gridboxboundaries_binary(gridfile, zgrid, xgrid, ygrid, constsfile)
rgrid.print_domain_info(constsfile, gridfile)

### ----- write thermodynamics binaries ----- ###
thermodyngen = gen_input_thermo.TimeVarying3DThermo(PRESSz0, TEMPz0, qvapz0, qcondz0,
                                                    WMAX, Zlength, Xlength, VMAX, Ylength)
cthermo.write_thermodynamics_binary(thermofile, thermodyngen, configfile,
                                    constsfile, gridfile)

### ----- write initial superdroplets binary ----- ###
nsupers = crdgens.nsupers_at_domain_base(gridfile, constsfile, npergbx, zlim)
radiigen  =  rgens.MonoAttrGen(monor)                 # all SDs have the same radius [m]
dryradiigen =  dryrgens.ScaledRadiiGen(dryr_sf)       # dryradii are 1/sf of radii [m]
coord3gen =  crdgens.SampleCoordGen(randcoord)        # (not) random coord3 of SDs
coord1gen =  crdgens.SampleCoordGen(randcoord)        # (not) random coord1 of SDs
coord2gen =  crdgens.SampleCoordGen(randcoord)        # (not) random coord2 of SDs
xiprobdist = probdists.DiracDelta(monor)              # monodisperse droplet probability distrib

initattrsgen = attrsgen.AttrsGenerator(radiigen, dryradiigen, xiprobdist,
                                       coord3gen, coord1gen, coord2gen)
csupers.write_initsuperdrops_binary(initSDsfile, initattrsgen,
                                    configfile, constsfile,
                                    gridfile, nsupers, numconc)

### ----- show (and save) plots of binary file data ----- ###
if isfigures[0]:
    rgrid.plot_gridboxboundaries(constsfile, gridfile,
                                 savefigpath, isfigures[1])
    rthermo.plot_thermodynamics(constsfile, configfile, gridfile,
                                thermofile, savefigpath, isfigures[1])
    plt.close()
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###

### ---------------------------------------------------------------- ###
### ---------------------- RUN CLEO EXECUTABLE --------------------- ###
### ---------------------------------------------------------------- ###
os.chdir(path2build)
os.system('pwd')
os.system('rm -rf '+dataset) # delete any existing dataset
executable = path2build+'/examples/yac/fromfile/src/yac1'
print('Executable: '+executable)
print('Config file: '+configfile)
os.system(executable + ' ' + configfile)
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###


### ---------------------------------------------------------------- ###
### ------------------------- PLOT RESULTS ------------------------- ###
### ---------------------------------------------------------------- ###
# read in constants and intial setup from setup .txt file
config = pysetuptxt.get_config(setupfile, nattrs=3, isprint=True)
consts = pysetuptxt.get_consts(setupfile, isprint=True)
gbxs = pygbxsdat.get_gridboxes(gridfile, consts["COORD0"], isprint=True)

time = pyzarr.get_time(dataset)
sddata = pyzarr.get_supers(dataset, consts)
totnsupers = pyzarr.get_totnsupers(dataset)
thermo, winds = pyzarr.get_thermodata(dataset, config["ntime"], gbxs["ndims"],
                                      consts, getwinds=True)

# plot super-droplet results
savename = savefigpath + "yac1_totnsupers_validation.png"
pltmoms.plot_totnsupers(time, totnsupers, savename=savename)

nsample = 1000
savename = savefigpath + "yac1_motion2d_validation.png"
pltsds.plot_randomsample_superdrops_2dmotion(sddata,
                                             config["totnsupers"],
                                             nsample,
                                             savename=savename,
                                             arrows=False,
                                             israndom=False)

# plot thermodynamics results
plot_output_thermo.plot_domain_thermodynamics_timeseries(time, gbxs, thermo, winds,
                                                         savedir=savefigpath)

### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
