""" 

Example 2: 

The example 2 for TurbSim has the same inputs as example 1 (the ones for the discretization), 
but additional inputs are now added to define the simulation setup. The new inputs are listed below:

- **General parameters**
    - `path` : folder (preferably new) where all the simulation files will be written
    -  `ffbin` and `tsbin`: location of the FAST.Farm and TurbSim executables
    - `templatePath`: existing folder, where the template FAST.Farm and OpenFAST files can be found.
    - `templateFiles`: location of files within templatePath to be used, or absolute location of required files.
        Note: some template files provided (like ED, SD, turbine fst) need to end with `.T` while  the actual filename inside `templatePath` is `*.T.<ext>` where `<ext>` is either `dat` or `fst`.

- **Farm parameters** (same as above)

- Inflow conditions and input files (same as above, but with more details):
   - `shear`, `TIvalue`, `inflow_deg`, together with `vhub` define the inflow values.
   - `tmax`: defines the maximum simulation time

- **Discretization parameters** (`dt_high`, `dt_low`, etc): these are the parameters we determined in Example 1.


### Pre-requisites to run all of the the cells below:
 - A FAST.Farm Executable with version v4.*  (to be compatible with the input files provided)
 - A TurbSim Executable (any version>2 should do)
 - ROSCO libdiscon DLL or shared object with version 4.9



This is a non-exaustive example. Additional details are available on the
docstrings of the FFCaseCreation constructor.

The extent of the high res and low res domain are setup according to the guidelines:
    https://openfast.readthedocs.io/en/dev/source/user/fast.farm/ModelGuidance.html

NOTE: This is an example using TurbSim inflow, so the resulting boxes are necessary to
      build the final FAST.Farm case and are not provided as part of this repository. 

"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openfast_toolbox.fastfarm.FASTFarmCaseCreation import FFCaseCreation

from openfast_toolbox.fastfarm.FASTFarmCaseCreation import check_files_exist, check_discon_library, modifyProperty
from openfast_toolbox.fastfarm.fastfarm import plotFastFarmSetup

scriptDir = os.path.dirname(__file__)


def main(test=False):

    # -----------------------------------------------------------------------------
    # USER INPUT: Modify these
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # ------------------------ General parameters ---------------------------------
    # -----------------------------------------------------------------------------

    path = os.path.join(scriptDir, '_ex2') # folder (preferably new) where all the simulation files will be written

    # ----------- Execution parameters
    # If you are sure the correct binaries are first on your $PATH and associated
    # libraries on $LD_LIBRARY_PATH, you can set the variables below to None or
    # remove them from the `FFCaseCreation` call
    ffbin     = os.path.join(scriptDir, './SampleFiles/dummy_fastfarm.exe.txt') # relative or absolute path of FAST.Farm executable
    tsbin     = os.path.join(scriptDir, './SampleFiles/dummy_turbsim.exe.txt')  # relative or absolute path of TurbSim executable
    libdiscon = os.path.join(scriptDir, './SampleFiles/dummy_discon.dll.dummy')  # relative or absolute path to discon shared library

    # -----------------------------------------------------------------------------
    # --------------------------- Farm parameters ---------------------------------
    # -----------------------------------------------------------------------------
    # ----------- General turbine parameters
    cmax     = 5      # Maximum blade chord (m), affects dr
    fmax     = 10/6   # Maximum excitation frequency (Hz), affects dt_high
    Cmeander = 1.9    # Meandering constant (-)
    D = 240           # Rotor diameter (m)
    zhub = 150        # Hub height (m)
    # ----------- Wind farm
    # The wts dictionary holds information of each wind turbine. The allowed entries
    # are: x, y, z, D, zhub, cmax, fmax, Cmeander, and phi_deg. The phi_deg is the
    # only entry that is optional and is related to floating platform heading angle,
    # given in degrees. The angle phi_deg is not illustrated on the example below.
    wts = {
        0 :{'x': 800.0, 'y': 480, 'z':0.0, 'D':D, 'zhub':zhub, 'cmax':cmax, 'fmax':fmax, 'Cmeander':Cmeander, 'name':'T1'},
        1 :{'x':1800.0, 'y': 480, 'z':0.0, 'D':D, 'zhub':zhub, 'cmax':cmax, 'fmax':fmax, 'Cmeander':Cmeander, 'name':'T2'},
        2 :{'x':2800.0, 'y': 480, 'z':0.0, 'D':D, 'zhub':zhub, 'cmax':cmax, 'fmax':fmax, 'Cmeander':Cmeander, 'name':'T3'},
        3 :{'x': 800.0, 'y':1440, 'z':0.0, 'D':D, 'zhub':zhub, 'cmax':cmax, 'fmax':fmax, 'Cmeander':Cmeander, 'name':'T4'},
        4 :{'x':1800.0, 'y':1440, 'z':0.0, 'D':D, 'zhub':zhub, 'cmax':cmax, 'fmax':fmax, 'Cmeander':Cmeander, 'name':'T5'},
        5 :{'x':2800.0, 'y':1440, 'z':0.0, 'D':D, 'zhub':zhub, 'cmax':cmax, 'fmax':fmax, 'Cmeander':Cmeander, 'name':'T6'},
    }
    refTurb_rot = 0

    # ----------- Turbine parameters
    # Set the yaw of each turbine for wind dir. One row for each wind direction.
    yaw_init = None


    # -----------------------------------------------------------------------------
    # ------------------- Inflow conditions and input files -----------------------
    # -----------------------------------------------------------------------------
    # ----------- Additional variables
    tmax = 120      # Total simulation time
    zbot = 1        # Bottom of your domain
    mod_wake = 2    # Wake model. 1: Polar, 2: Curled, 3: Cartesian
    # ----------- Inflow parameters
    inflowType = 'TS'
    # ----------- Desired sweeps, fill array with multiple values if necessary
    nSeeds      = 6     # Number of seeds
    vhub       = [8]    # Hub velocity [m/s]
    shear      = [0.1]  # Power law exponent [-]
    TIvalue    = [10]   # Turbulence intensity [%]
    inflow_deg = [0]    # Wind direction [deg]

    # ----------- Template files
    # --- Option 1
    templateFSTF = os.path.join(scriptDir, '../../../data/IEA15MW/FF.fstf')
    templateFiles = {'libdisconfilepath' : libdiscon}    
    # --- Option 2
    #templatePath = '/full/path/where/template/files/are'
    ## Files should be in templatePath. Put None on any input that is not applicable.
    #templateFiles = {
    #    "EDfilename"              : 'ElastoDyn.T',
    #    'SEDfilename'             : None,  # 'SimplifiedElastoDyn.T',
    #    'HDfilename'              : None,  # 'HydroDyn.dat', # ending with .T for per-turbine HD, .dat for holisitc
    #    'MDfilename'              : None,  # 'MoorDyn.T',    # ending with .T for per-turbine MD, .dat for holistic
    #    'SSfilename'              : None,  # 'SeaState.dat',
    #    'SrvDfilename'            : 'ServoDyn.T',
    #    'ADfilename'              : 'AeroDyn.dat',
    #    'ADskfilename'            : None,
    #    'SubDfilename'            : 'SubDyn.dat',
    #    'IWfilename'              : 'InflowWind.dat',
    #    'BDfilename'              : None,
    #    'EDbladefilename'         : 'ElastoDyn_Blade.dat',
    #    'EDtowerfilename'         : 'ElastoDyn_Tower.dat',
    #    'ADbladefilename'         : 'AeroDyn_Blade.dat',
    #    'turbfilename'            : 'Model.T',
    #    'libdisconfilepath'       : '/full/path/to/controller/libdiscon.so',
    #    'controllerInputfilename' : 'DISCON.IN',
    #    'coeffTablefilename'      : None,
    #    'hydroDatapath'           : None,  # '/full/path/to/hydroData',
    #    'FFfilename'              : 'Model_FFarm.fstf',
    #    'turbsimLowfilepath'      : './SampleFiles/template_Low_InflowXX_SeedY.inp',
    #    'turbsimHighfilepath'     : './SampleFiles/template_HighT1_InflowXX_SeedY.inp'
    #}
    # SLURM scripts
    slurm_TS_high           = os.path.join(scriptDir, './SampleFiles/runAllHighBox.sh')
    slurm_TS_low            = os.path.join(scriptDir, './SampleFiles/runAllLowBox.sh')
    slurm_FF_single         = './SampleFiles/runFASTFarm_cond0_case0_seed0.sh'


    # -----------------------------------------------------------------------------
    # ---------------------- Discretization parameters ----------------------------
    # -----------------------------------------------------------------------------
    # The values below are illustrative and do not represent recommended values.
    # The user should set these according to the farm and computational resources
    # available. For the recommended temporal and spatial resolutions, follow the
    # modeling guidances or the example `Ex1_FASTFarm_discretization.py`.

    # ----------- Low- and high-res boxes parameters
    # High-res boxes settings
    dt_high     =  0.5                # sampling time of high-res files [s]
    ds_high     =  5                  # dx, dy, dz of high-res files [m]
    extent_high =  1.2                # extent in y and x for each turbine, in D
    # Low-res boxes settings
    dt_low      = 4                   # sampling time of low-res files [s]
    ds_low      = 10                  # dx, dy, dz of low-res files [m]
    extent_low  = [1.5,2.5,1.5,1.5,2] # extent in [xmin,xmax,ymin,ymax,zmax], in D


    # -----------------------------------------------------------------------------
    # END OF USER INPUT
    # -----------------------------------------------------------------------------

    # Plot turbines locations
    fig = plotFastFarmSetup(wts)
    if not test:
        check_files_exist(ffbin, tsbin, templateFSTF, templateFiles)
        check_discon_library(libdiscon)

    # -----------------------------------------------------------------------------
    # -------------------- FAST.Farm initial setup --------------------------------
    # -----------------------------------------------------------------------------
    # ----------- Initial setup
    ffcase = FFCaseCreation(path, wts, tmax, zbot, vhub, shear, TIvalue, inflow_deg,
                            dt_high=dt_high, ds_high=ds_high, extent_high=extent_high,
                            dt_low=dt_low,   ds_low=ds_low,   extent_low=extent_low,
                            ffbin=ffbin, mod_wake=mod_wake, yaw_init=yaw_init,
                            nSeeds=nSeeds, tsbin=tsbin, inflowType=inflowType,
                            refTurb_rot=refTurb_rot, verbose=0)

    # ----------- Perform auxiliary steps in preparing the case
    ffcase.setTemplateFilename(templateFiles=templateFiles, templateFSTF=templateFSTF) # Option 1
    #ffcase.setTemplateFilename(templatePath, templateFiles) # Option 2
    ffcase.getDomainParameters()
    ffcase.copyTurbineFilesForEachCase()
    ffcase.plot()  # add showTurbNumber=True to help with potential debugging


    # -----------------------------------------------------------------------------
    # ---------------------- TurbSim setup and execution --------------------------
    # -----------------------------------------------------------------------------
    # ----------- TurbSim low-res setup
    ffcase.TS_low_setup() # Create TurbSim input files for low res
    # ----------- Prepare script for submission
    ffcase.TS_low_batch_prepare()
    #ffcase.TS_low_slurm_prepare(slurm_TS_low)
    # ----------- Submit the low-res script (can be done from the command line)
    #ffcase.TS_low_batch_run() # Write a batch file to disk
    #ffcase.TS_low_slurm_submit() # Alternative, write a slurm batch file, see below

    # The low-resolution boxes need to be executed before we can proceed setting up
    # the high-resolution and the overall FAST.Farm case. The lines below will need
    # to be executed one at a time, pending the successful completion of Turbsim.

    if test:
        # We have to stop here
        return ffcase 

    # ----------- TurbSim high-res setup
    ffcase.TS_high_setup() # Create TurbSim input files
    # ----------- Prepare script for submission
    ffcase.TS_high_batch_prepare() # Write a batch file to disk
    #ffcase.TS_high_slurm_prepare(slurm_TS_high) # Alternative, write a slurm batch file
    # ----------- Submit the high-res script (can be done from the command line)
    ffcase.TS_high_batch_run(showOutputs=True, showCommand=True, nBuffer=8, shell_cmd='bash')
    #ffcase.TS_high_slurm_submit() # Alternative, submit a slurm batch file


    # -----------------------------------------------------------------------------
    # ------------------ Finish FAST.Farm setup and execution ---------------------
    # -----------------------------------------------------------------------------
    # ----------- FAST.Farm setup
    ffcase.FF_setup() # Write FAST.Farm input files
    # Update wake model constants (adjust as needed for your turbine model)
    ffcase.set_wake_model_params(k_VortexDecay=0, k_vCurl=2.8)

    # ----------- Prepare script for submission
    ffcase.FF_batch_prepare() # Write batch files with all commands to be run
    #ffcase.FF_slurm_prepare(slurm_FF_single) # Alternative, prepare a slurm batch file

    # We can do simple modifications:
    modifyProperty(ffcase.FFFiles[0], 'NX_Low', 100) # Making the domain longer for visualization purposes
    # We can visualize the setup:
    plotFastFarmSetup(ffcase.FFFiles[0], grid=True, figsize=(10,3));


    # ----------- Submit the FAST.Farm script (can be done from the command line)
    ffcase.FF_batch_run(showOutputs=True, showCommand=True, nBuffer=10, shell_cmd='bash')
    #ffcase.FF_slurm_submit(p='debug', t='1:00:00') # Alternative, submit a slurm batch file
    return ffcase


if __name__ == '__main__':
    # This example cannot be fully run.
    ffcase = main(test=True)



if __name__=='__test__':
    # This example cannot be fully run.
    ffcase = main(test=True)
    from openfast_toolbox.io import FASTInputFile
    import shutil

    np.testing.assert_(os.path.exists(ffcase.path), f"path does not exist {ffcase.path}")
    np.testing.assert_equal(ffcase.nConditions, 1)
    np.testing.assert_equal(ffcase.nCases, 1)
    np.testing.assert_equal(ffcase.nSeeds, 6)
    np.testing.assert_equal(ffcase.condDirList, ['Cond00_v08.0_PL0.1_TI10'])
    for cond, _ in enumerate(ffcase.condDirList):
        np.testing.assert_(os.path.exists(ffcase.getCondPath(cond)), f"cond path {cond} does not exist ")
        for seed in range(ffcase.nSeeds):
            seedPath = ffcase.getCondSeedPath(cond, seed)
            np.testing.assert_(os.path.exists(ffcase.getCondSeedPath(cond, seed)), f"cond seed path {cond} {seed} does not exist ")
            # Try to read TS input file
            tsinp = os.path.join(seedPath, 'Low.inp')
            ts = FASTInputFile(tsinp)

        for case in range(ffcase.nCases):
            np.testing.assert_(os.path.exists(ffcase.getCasePath(cond, case)), f"case path {cond} {case} does not exist ")
            for seed in range(ffcase.nSeeds):
                np.testing.assert_(os.path.exists(ffcase.getCaseSeedPath(cond, case, seed)), f"case seed path {cond} {case} {seed} does not exist ")
    # TODO add more tests
    if os.path.exists(ffcase.path):
        try:
            shutil.rmtree(ffcase.path)
        except OSError as e:
            print('Fail to remove FAST.Farm ex2 folder' )
            pass
