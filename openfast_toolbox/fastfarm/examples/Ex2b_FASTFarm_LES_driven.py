""" 
Setup a FAST.Farm suite of cases based on input parameters. Uses LES inflow.
This is a non-exaustive example. Additional details are available on the
docstrings of the FFCaseCreation constructor.

The extent of the high res and low res domain are setup according to the guidelines:
    https://openfast.readthedocs.io/en/dev/source/user/fast.farm/ModelGuidance.html

"""

from openfast_toolbox.fastfarm.FASTFarmCaseCreation import FFCaseCreation
from openfast_toolbox.fastfarm.AMRWindSimulation import AMRWindSimulation

def main():

    # -----------------------------------------------------------------------------
    # USER INPUT: Modify these
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # ------------------------ General parameters ---------------------------------
    # -----------------------------------------------------------------------------

    # ----------- Case absolute path
    path = '/complete/path/of/your/case'

    # ----------- Execution parameters
    # If you are sure the correct binary is first on your $PATH and associated
    # libraries on $LD_LIBRARY_PATH, you can set the variable below to None or
    # remove it from the `FFCaseCreation` call
    ffbin = '/full/path/to/your/binary/.../bin/FAST.Farm'


    # -----------------------------------------------------------------------------
    # --------------------------- Farm parameters ---------------------------------
    # -----------------------------------------------------------------------------

    # ----------- General turbine parameters
    cmax     = 5      # Maximum blade chord (m)
    fmax     = 10/6   # Maximum excitation frequency (Hz)
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
    vhub       = [8]
    shear      = [1]
    TIvalue    = [1]
    inflow_deg = [0]

    # ----------- Template files
    templatePath = '/full/path/where/template/files/are'
    # Files should be in templatePath. Put None on any input that is not applicable.
    templateFiles = {
        "EDfilename"              : 'ElastoDyn.T',
        'SEDfilename'             : None,  # 'SimplifiedElastoDyn.T',
        'HDfilename'              : None,  # 'HydroDyn.dat', # ending with .T for per-turbine HD, .dat for holisitc
        'MDfilename'              : None,  # 'MoorDyn.T',    # ending with .T for per-turbine MD, .dat for holistic
        'SSfilename'              : None,  # 'SeaState.dat',
        'SrvDfilename'            : 'ServoDyn.T',
        'ADfilename'              : 'AeroDyn.dat',
        'ADskfilename'            : None,
        'SubDfilename'            : 'SubDyn.dat',
        'IWfilename'              : 'InflowWind.dat',
        'BDfilename'              : None,
        'EDbladefilename'         : 'ElastoDyn_Blade.dat',
        'EDtowerfilename'         : 'ElastoDyn_Tower.dat',
        'ADbladefilename'         : 'AeroDyn_Blade.dat',
        'turbfilename'            : 'Model.T',
        'libdisconfilepath'       : '/full/path/to/controller/libdiscon.so',
        'controllerInputfilename' : 'DISCON.IN',
        'coeffTablefilename'      : None,
        'hydroDatapath'           : None,  # '/full/path/to/hydroData',
        'FFfilename'              : 'Model_FFarm.fstf',
    }
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
    amr.write_sampling_params(os.path.join(path,'FF_boxes.i'), overwrite=True)


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
    dt_high     = amr.dt_high_les  # sampling frequency of high-res files
    ds_high     = amr.ds_high_les  # dx, dy, dz of high-res files
    extent_high = amr.extent_high  # extent in y and x for each turbine, in D.
    # Low-res boxes settings
    dt_low      = amr.dt_low_les   # sampling frequency of low-res files
    ds_low      = amr.ds_low_les   # dx, dy, dz of low-res files
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
    ffcase.setTemplateFilename(templatePath, templateFiles)   
    ffcase.getDomainParameters()
    ffcase.copyTurbineFilesForEachCase()


    # -----------------------------------------------------------------------------
    # ------------------ Finish FAST.Farm setup and execution ---------------------
    # -----------------------------------------------------------------------------
    # ----------- FAST.Farm setup
    ffcase.FF_setup()
    # Update wake model constants (adjust as needed for your turbine model)
    ffcase.set_wake_model_params(k_VortexDecay=0, k_vCurl=2.8)

    # ----------- Prepare script for submission
    ffcase.FF_slurm_prepare(slurm_FF_single)

    # ----------- Submit the FAST.Farm script (can be done from the command line)
    ffcase.FF_slurm_submit(p='debug', t='1:00:00')



if __name__ == '__main__':
    # This example cannot be fully run.
    pass
