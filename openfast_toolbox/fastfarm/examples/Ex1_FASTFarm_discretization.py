""" 
Setup a FAST.Farm suite of cases based on input parameters. Uses TurbSim inflow.
This is a non-exaustive example. Additional details are available on the
docstrings of the FFCaseCreation constructor.

The extent of the high res and low res domain are setup according to the guidelines:
    https://openfast.readthedocs.io/en/dev/source/user/fast.farm/ModelGuidance.html

NOTE: This is an example using TurbSim inflow, so the resulting boxes are necessary to
      build the final FAST.Farm case and are not provided as part of this repository. 

"""

from openfast_toolbox.fastfarm.FASTFarmCaseCreation import FFCaseCreation

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
    # If you are sure the correct binaries are first on your $PATH and associated
    # libraries on $LD_LIBRARY_PATH, you can set the variables below to None or
    # remove them from the `FFCaseCreation` call
    ffbin = '/full/path/to/your/binary/.../bin/FAST.Farm'
    tsbin = '/full/path/to/your/binary/.../bin/turbsim'


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
    tmax = 120      # Total simulation time
    nSeeds = 6      # Number of seeds
    zbot = 1        # Bottom of your domain
    mod_wake = 2    # Wake model. 1: Polar, 2: Curled, 3: Cartesian

    # ----------- Inflow parameters
    inflowType = 'TS'

    # ----------- Desired sweeps
    vhub       = [8]
    shear      = [0.1]
    TIvalue    = [10]
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
        'turbsimLowfilepath'      : './SampleFiles/template_Low_InflowXX_SeedY.inp',
        'turbsimHighfilepath'     : './SampleFiles/template_HighT1_InflowXX_SeedY.inp'
    }
    # SLURM scripts
    slurm_TS_high           = './SampleFiles/runAllHighBox.sh'
    slurm_TS_low            = './SampleFiles/runAllLowBox.sh'
    slurm_FF_single         = './SampleFiles/runFASTFarm_cond0_case0_seed0.sh'


    # -----------------------------------------------------------------------------
    # ---------------------- Discretization parameters ----------------------------
    # -----------------------------------------------------------------------------
    # Let's start with no knowledge of what resolution we should use.

    # ----------- Low- and high-res boxes parameters
    # High-res boxes settings
    dt_high     = None    # sampling frequency of high-res files
    ds_high     = None    # dx, dy, dz of high-res files
    extent_high = None   # extent in y and x for each turbine, in D
    # Low-res boxes settings
    dt_low      = None   # sampling frequency of low-res files
    ds_low      = None   # dx, dy, dz of low-res files
    extent_low  = None   # extent in [xmin,xmax,ymin,ymax,zmax], in D


    # -----------------------------------------------------------------------------
    # END OF USER INPUT
    # -----------------------------------------------------------------------------



    # -----------------------------------------------------------------------------
    # -------------------- FAST.Farm initial setup --------------------------------
    # -----------------------------------------------------------------------------
    # ----------- Initial setup
    # The initial setup with the temporal and spatial resolutions of both high and
    # low resolution boxes given as None will trigger their automatic computation.
    amr = FFCaseCreation(path, wts, tmax, zbot, vhub, shear, TIvalue, inflow_deg,
                         dt_high=dt_high, ds_high=ds_high, extent_high=extent_high,
                         dt_low=dt_low,   ds_low=ds_low,   extent_low=extent_low,
                         ffbin=ffbin, mod_wake=mod_wake, yaw_init=yaw_init,
                         nSeeds=nSeeds, tsbin=tsbin, inflowType=inflowType,
                         refTurb_rot=refTurb_rot, verbose=0)



if __name__ == '__main__':
    # This example cannot be fully run.
    pass
