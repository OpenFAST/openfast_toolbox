import numpy as np
import os
import matplotlib.pyplot as plt

import openfast_toolbox.case_generation.case_gen as case_gen
import openfast_toolbox.postpro as postpro

# Get current directory so this script can be called from any location
MyDir=os.path.dirname(__file__)

def CPLambdaExample():
    """ Example to determine the CP-CT Lambda Pitch matrices of a turbine.
    This script uses the function CPCT_LambdaPitch which basically does the same as Parametric Examples given in this folder
    """
    FAST_EXE  = os.path.join(MyDir, '../../../data/openfast.exe') # Location of a FAST exe (and dll)
    ref_dir   = os.path.join(MyDir, '../../../data/NREL5MW/')     # Folder where the fast input files are located (will be copied)
    main_file = 'Main_Onshore.fst'  # Main file in ref_dir, used as a template

    # --- Computing CP and CT matrices for range of lambda and pitches
    Lambda = np.linspace(0.1,10,3)
    Pitch  = np.linspace(-10,10,4)

    # We return a ROSCOPerformanceFile, "rs"
    rs, result, _ = case_gen.CPCT_LambdaPitch(ref_dir, main_file, Lambda, Pitch, fastExe=FAST_EXE, showOutputs=False, nCores=4, TMax=10)

    # --- Methods of ROSCOPerformanceFile
    # see openfast_toolbox/io/ROSCOPerformanceFile.py

    # Get values at CPmax
    CPmax, tsr_max, pitch_max = rs.CPmax()
    print('CP max',CPmax)

    # Plot 3D figure
    fig = rs.plotCP3D(plotMax=True)

    # Get vectors / arrays from parametric study
    #pitch = rs['pitch'] 
    #tsr   = rs['TSR']   
    #WS    = rs['WS']    
    #CP    = rs['CP']    
    #CT    = rs['CT']    
    #CQ    = rs['CQ']    

if __name__=='__main__':
    CPLambdaExample()
    plt.show()
if __name__=='__test__':
    # Need openfast.exe, doing nothing
    pass
