""" 
1. Example 1: 
 This example calculates the desired temporal and spatial resolution given a wind farm layout.
 Assumes that we use TurbSim inflow. 

 The FAST.Farm guidelines requires several parameters to be set to find the resolutsion:
  - Spatial parameters: max chord (`cmax`), rotor diameter (`D`), meandering constant (`Cmeander`)
  - Temporal parameters: maximum excitation frequency (`fmax`), mean wind speed (`vhub`)
  - Model parameters: wake models (`mod_wake`), and background inflow type (`inflowType`)
 
 Based on these parameters, the FFCaseCreation class can compute some default resolution, but it is often required to adjust some of them manually and not fully rely on the defaults.
 
 In this example, we do the following:
 - First, we obtain the default parameters and plot the layout
 - Then, we manually adjust some of the resolution parameters



This is a non-exaustive example. Additional details are available on the
docstrings of the FFCaseCreation constructor.

The extent of the high res and low res domain are setup according to the guidelines:
    https://openfast.readthedocs.io/en/dev/source/user/fast.farm/ModelGuidance.html

NOTE: This is an example using TurbSim inflow, so the resulting boxes are necessary to
      build the final FAST.Farm case and are not provided as part of this repository. 

"""
import os
import numpy as np
from openfast_toolbox.fastfarm.FASTFarmCaseCreation import FFCaseCreation

# Get current directory so this script can be called from any location
scriptDir = os.path.dirname(__file__)

def main(test=False):

    # -----------------------------------------------------------------------------
    # USER INPUT: Modify these
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # ------------------------ General parameters ---------------------------------
    # -----------------------------------------------------------------------------

    # ----------- Case absolute path
    path = os.path.join(scriptDir, '_ex1') # folder (preferably new) where all the simulation files will be written

    # ----------- Execution parameters


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


    # -----------------------------------------------------------------------------
    # ---------------------- Discretization parameters ----------------------------
    # -----------------------------------------------------------------------------
    # Let's start with no knowledge of what resolution we should use.

    # ----------- Low- and high-res boxes parameters
    # High-res boxes settings
    dt_high     = None   # sampling time of high-res files [s]
    ds_high     = None   # dx, dy, dz of high-res files [m]
    extent_high = None   # extent in y and x for each turbine, in D
    # Low-res boxes settings
    dt_low      = None   # sampling time of low-res files [s]
    ds_low      = None   # dx, dy, dz of low-res files [m]
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
    # --- Generic call
    #ffcase = FFCaseCreation(path, wts, tmax, zbot, vhub, shear, TIvalue, inflow_deg,
    #                     dt_high=dt_high, ds_high=ds_high, extent_high=extent_high,
    #                     dt_low=dt_low,   ds_low=ds_low,   extent_low=extent_low,
    #                     ffbin=ffbin, mod_wake=mod_wake, yaw_init=yaw_init,
    #                     nSeeds=nSeeds, tsbin=tsbin, inflowType=inflowType,
    #                     refTurb_rot=refTurb_rot, verbose=0)
    # --------------------------------------------------------------------------------
    # --- 1.2 Getting the default resolution and plotting the layout
    # --------------------------------------------------------------------------------
    # Below we provide the minimal set of arguments needed to compute the resolution automatically.
    ffcase = FFCaseCreation(path=path, wts=wts, vhub=vhub, 
                          mod_wake=mod_wake,
                          inflowType=inflowType)

    # Plot the FARM Layout
    ffcase.plot() # Note: the grids are not plotted by this function

    # --------------------------------------------------------------------------------
    # --- 1.3 Adjusting the resolution
    # --------------------------------------------------------------------------------
    # The output message above is saying that the low-res should be 24.32, but since it needs to be a multiple of the high res, then it automatically selects 20 m. However, since 24.32 is close to 25, we might want to just run at 25. Then, an option is to re-run the same command, but now passing a value for `ds_low`:
    # --- Case 2 Prescribing some of the values
    dt_high = 0.50 # [s]
    dt_low  = 2.00 # [s]
    ds_low  = 25   # [m]
    ffcase2 = FFCaseCreation(path=path, wts=wts, vhub=vhub,
                         dt_high=dt_high,
                         dt_low=dt_low,   ds_low=ds_low,
                         mod_wake=mod_wake,
                         inflowType=inflowType, verbose=0)

    """
     This ends the illustration of the first example. Now we can move forward with the FAST.Farm setup using two options:
     
     1. Use directly the `ffcase` object:
     ```
     # ----------- Low- and high-res boxes parameters
     # High-res boxes settings
     dt_high     = ffcase.dt_high
     ds_high     = ffcase.ds_high
     extent_high = ffcase.extent_high
     # Low-res boxes settings
     dt_low      = ffcase.dt_low
     ds_low      = ffcase.ds_low
     extent_low  = ffcase.extent_low
     ```
     
     2. Manually add those values to their corresponding variables:
     ```
     # ----------- Low- and high-res boxes parameters
     # High-res boxes settings
     dt_high     =  0.5               
     ds_high     =  5                 
     extent_high =  1.2               
     # Low-res boxes settings
     dt_low      = 1.0                  
     ds_low      = 25                 
     extent_low  = [1.5,2.5,1.5,1.5,2]
     
    """

    return ffcase, ffcase2


if __name__ == '__main__':
    ffcase, ffcase2 = main(test=False)
    # Note you can always print the object and get some information about the farm and the set of cases that will be setup:
    print(ffcase)

if __name__=='__test__':
    ffcase, ffcase2 = main(test=True)
    np.testing.assert_equal(ffcase.ds_low, 20)
    np.testing.assert_equal(ffcase.dt_low, 0.9)
    np.testing.assert_equal(ffcase.ds_high, 5)
    np.testing.assert_equal(ffcase.dt_high, 0.3)
    np.testing.assert_array_equal(ffcase.extent_low, [3, 6, 3, 3, 2] )
    np.testing.assert_equal(ffcase.vhub    , [8])
    np.testing.assert_equal(ffcase.inflowType, 'TS')
    # NOTE: these shouldn't matter:
    np.testing.assert_equal(ffcase.shear   , [0])
    np.testing.assert_equal(ffcase.TIvalue, [10])

    np.testing.assert_equal(ffcase2.ds_low, 25)
    np.testing.assert_equal(ffcase2.dt_low, 2.0)
    np.testing.assert_equal(ffcase2.ds_high, 5)
    np.testing.assert_equal(ffcase2.dt_high, 0.5)
    np.testing.assert_array_equal(ffcase2.extent_low, [3, 6, 3, 3, 2] )


