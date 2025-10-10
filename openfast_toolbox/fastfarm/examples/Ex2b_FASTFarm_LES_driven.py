""" 
Setup a FAST.Farm suite of cases based on input parameters. Uses LES inflow.
This is a non-exaustive example. Additional details are available on the
docstrings of the FFCaseCreation constructor.

The extent of the high res and low res domain are setup according to the guidelines:
    https://openfast.readthedocs.io/en/dev/source/user/fast.farm/ModelGuidance.html

"""
import os
from openfast_toolbox.fastfarm.FASTFarmCaseCreation import FFCaseCreation
from openfast_toolbox.fastfarm.AMRWindSimulation import AMRWindSimulation


scriptDir = os.path.dirname(__file__)

def main(test=False):

    # -----------------------------------------------------------------------------
    # USER INPUT: Modify these
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # ------------------------ General parameters ---------------------------------
    # -----------------------------------------------------------------------------

    # ----------- Case absolute path
    path = os.path.join(scriptDir, '_ex3') # folder (preferably new) where all the simulation files will be written

    # ----------- Execution parameters
    # If you are sure the correct binary is first on your $PATH and associated
    # libraries on $LD_LIBRARY_PATH, you can set the variable below to None or
    # remove it from the `FFCaseCreation` call
    ffbin     = os.path.join(scriptDir, './SampleFiles/dummy_fastfarm.exe.txt') # relative or absolute path of FAST.Farm executable

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
    tmax = 60       # Total simulation time
    zbot = 1        # Bottom of your domain
    mod_wake = 2    # Wake model. 1: Polar, 2: Curled, 3: Cartesian
    # ----------- Inflow parameters
    inflowType = 'LES'
    inflowPath = '/full/path/to/LES/case/.../LESboxes'

    # ----------- Desired sweeps
    # These sweeps are not used for LES, and are here for generality with cases
    # driven by TurbSim. They are only used on the directory names. If multiple
    # executions with multiple different LES solutions, you can give inflowPath
    # as a n-sized array (related to the different LES), and give the variables
    # below as n-sized arrays as well.
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
    #}
    # SLURM scripts
    slurm_FF_single         = './SampleFiles/runFASTFarm_cond0_case0_seed0.sh'


    # -----------------------------------------------------------------------------
    # ------------------------------- LES parameters ------------------------------
    # -----------------------------------------------------------------------------
    # The values below are illustrative and do not represent recommended values.
    # The user should set the LES grid parameters according to the farm and the
    # available computational resources.
    # ----------- AMR-Wind parameters
    fixed_dt = 0.1
    prob_lo =  (0.0, 0.0, 0.0)
    prob_hi =  (3840, 1920, 960) 
    n_cell  =  (384, 192, 96)
    max_level = 3  # Number of grid refinement levels
    incflo_velocity_hh = (8, 0, 0) 
    postproc_name = 'box'

    # ----------- Low- and high-res boxes buffer parameters
    buffer_lr = [1.5,2.5,1.5,1.5,2]
    buffer_hr = 0.6

    # ----------- Levels of each box
    level_lr=0
    level_hr=3

    # -----------Create AMR-Wind sampling parameters
    # The call below should be done in an iterative fashion. First set the temporal
    # and spatial resolutions to None and see the output. The user should adjust as
    # needed by explicitly passing ds_hr, ds_lr, dt_hr, and dt_lr on another call, 
    # constrained by the analysis  goal and the available computational resources.
    amr = AMRWindSimulation(wts, fixed_dt, prob_lo, prob_hi,
                           n_cell, max_level, incflo_velocity_hh,
                           postproc_name, buffer_lr, buffer_hr,
                           ds_hr=None, ds_lr=None,
                           dt_hr=None, dt_lr=None,
                           mod_wake = mod_wake,
                           level_lr=level_lr, level_hr=level_hr, verbose=0)

    # -----------Save AMR-Wind sampling input
    amr.write_sampling_params(os.path.join(path,'FF_boxes.i'), overwrite=True, terrain=False)


    # -----------------------------------------------------------------------------
    # ---------------------- Discretization parameters ----------------------------
    # -----------------------------------------------------------------------------
    # The values below are illustrative and do not represent recommended values.
    # The user should set these according to the farm and computational resources
    # available. For the recommended temporal and spatial resolutions, follow the
    # modeling guidances or the example `Ex1_FASTFarm_discretization.py`. Note that
    # these values should match the sampling from the LES inflow generation step.

    # ----------- Low- and high-res boxes parameters
    # High-res boxes settings
    dt_high     = amr.dt_high_les  # sampling time of high-res files [s]
    ds_high     = amr.ds_high_les  # dx, dy, dz of high-res files [m]
    extent_high = amr.extent_high  # extent in y and x for each turbine, in D
    # Low-res boxes settings
    dt_low      = amr.dt_low_les   # sampling time of low-res files [s]
    ds_low      = amr.ds_low_les   # dx, dy, dz of low-res files [m]
    extent_low  = amr.extent_low   # extent in [xmin,xmax,ymin,ymax,zmax], in D


    # -----------------------------------------------------------------------------
    # END OF USER INPUT
    # -----------------------------------------------------------------------------



    # -----------------------------------------------------------------------------
    # -------------------- FAST.Farm initial setup --------------------------------
    # -----------------------------------------------------------------------------
    # ----------- Initial setup
    ffcase = FFCaseCreation(path, wts, tmax, zbot, vhub, shear, TIvalue, inflow_deg,
                            dt_high=dt_high, ds_high=ds_high, extent_high=extent_high,
                            dt_low=dt_low,   ds_low=ds_low,   extent_low=extent_low,
                            ffbin=ffbin, mod_wake=mod_wake, yaw_init=yaw_init,
                            inflowType=inflowType, inflowPath=inflowPath,
                            refTurb_rot=refTurb_rot, verbose=0)

    # ----------- Perform auxiliary steps in preparing the case
    ffcase.setTemplateFilename(templateFiles=templateFiles, templateFSTF=templateFSTF) # Option 1
    #ffcase.setTemplateFilename(templatePath, templateFiles) # Option 2
    ffcase.getDomainParameters()
    ffcase.copyTurbineFilesForEachCase()


    # -----------------------------------------------------------------------------
    # ------------------ Finish FAST.Farm setup and execution ---------------------
    # -----------------------------------------------------------------------------
    if test:
        return amr, ffcase
    # ----------- FAST.Farm setup
    ffcase.FF_setup() # Write FAST.Farm input files
    # Update wake model constants (adjust as needed for your turbine model)
    ffcase.set_wake_model_params(k_VortexDecay=0, k_vCurl=2.8)

    # ----------- Prepare script for submission
    ffcase.FF_batch_prepare() # Write batch files with all commands to be run
    #ffcase.FF_slurm_prepare(slurm_FF_single) # Alternative, prepare a slurm batch file

    # ----------- Submit the FAST.Farm script (can be done from the command line)
    ffcase.FF_batch_run(showOutputs=True, showCommand=True, nBuffer=10, shell_cmd='bash')
    #ffcase.FF_slurm_submit(p='debug', t='1:00:00') # Alternative, submit a slurm batch file
    return amr, ffcase


if __name__ == '__main__':
    amr, ffcase = main(test=True)
    # This example cannot be fully run.
    #print(amr)
    #print(ffcase)
    pass
if __name__ == '__test__':
    amr, ffcase = main(test=True)
    # This example cannot be fully run.
    import numpy as np
    import shutil
    np.testing.assert_equal(amr.ds_low_les, 20.0)
    np.testing.assert_equal(amr.dt_low_les, 0.9)
    np.testing.assert_array_equal(amr.extent_low, [1.5, 2.5, 1.5, 1.5, 2] )
    np.testing.assert_equal(amr.ds_high_les, 5.0)
    np.testing.assert_equal(amr.dt_high_les, 0.3)
    np.testing.assert_equal(ffcase.ds_low, 20.0)
    np.testing.assert_equal(ffcase.dt_low, 0.9)
    np.testing.assert_equal(ffcase.ds_high, 5.0)
    np.testing.assert_equal(ffcase.dt_high, 0.3)
    np.testing.assert_(os.path.exists(ffcase.path), f"path does not exist {ffcase.path}")
    for cond, _ in enumerate(ffcase.condDirList):
        np.testing.assert_(os.path.exists(ffcase.getCondPath(cond)), f"cond path {cond} does not exist ")
        for case in range(ffcase.nCases):
            np.testing.assert_(os.path.exists(ffcase.getCasePath(cond, case)), f"case path {cond} {case} does not exist ")
    if os.path.exists(ffcase.path):
        try:
            shutil.rmtree(ffcase.path)
        except OSError as e:
            print('Fail to remove FAST.Farm ex2 folder' )
            pass
