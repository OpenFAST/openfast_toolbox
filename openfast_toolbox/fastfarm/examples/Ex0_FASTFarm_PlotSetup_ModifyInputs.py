"""
Example 0: Modify and Plot Existing FASTFarm setup
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from openfast_toolbox.fastfarm.FASTFarmCaseCreation import FFCaseCreation # Main case creation class
from openfast_toolbox.fastfarm.fastfarm import plotFastFarmSetup # Low level FAST.Farm functions 
from openfast_toolbox.fastfarm.fastfarm import printWT
from openfast_toolbox.io.fast_input_file import FASTInputFile

# Get current directory so this script can be called from any location
scriptDir = os.path.dirname(__file__)

def main(test=False):

    FFfilepath = os.path.join(scriptDir, '../../../data/IEA15MW/FF.fstf')


    # --- 0.2 Read and Modify an existing FAST.Farm file
    # In this section, we look at the keys available in a FAST.Farm file, we modify some keys (to fix the issue with the low-res domain), and write to a new file.
    fstf = FASTInputFile(FFfilepath)

    if not test:
        plotFastFarmSetup(fstf, figsize=(10,3))
        print('Keys available in the input file: ', fstf.keys())
        # Let's look at the low resolution inputs:
        print('X0_Low:', fstf['X0_Low'])
        print('Y0_Low:', fstf['Y0_Low'])
        print('Z0_Low:', fstf['Z0_Low'])
        print('dX_Low:', fstf['dX_Low'])
        print('dY_Low:', fstf['dY_Low'])
        print('dZ_Low:', fstf['dZ_Low'])
        print('NX_Low:', fstf['NX_Low'])
        print('NY_Low:', fstf['NY_Low'])
        print('NZ_Low:', fstf['NZ_Low'])
        # We can estimate the y-extent:
        print('Y_low start : ', fstf['Y0_Low'])
        print('Y_low end   : ', fstf['Y0_Low']+fstf['dY_Low']*fstf['NY_Low'])
        print('Y_low extent: ', fstf['dY_Low']*fstf['NY_Low'])


    # --- Modify the low resolution inputs below so that it's centered about the y=0 line
    fstf['NY_Low'] = 41  # Can you find the value such that the low-res box is centered about the y=0 line?

    # --- 
    # Let's fix the bounding box of the high-res domain, which is defined using the `N*_High` keys and for each turbine within the key `WindTurbines`.
    if not test:
        print('WindTurbine Array from input file:\n',fstf['WindTurbines'],'\n')
        # Pretty print
        printWT(fstf)
        print('')
        print('NX_High:', fstf['NX_High'])
        print('NY_High:', fstf['NY_High'])
        print('NZ_High:', fstf['NZ_High'])

    # --- Which parameter should we change to make sure the turbine fits in the domain?
    #fstf['NZ_High'] = ?
    fstf['WindTurbines'][0, 9] = 50 # Fix this parameter

    plotFastFarmSetup(fstf, figsize=(10,3))

    # Once we have fixed it, we can write to a new input file:
    fstf.write('./_FF_new.fstf')

    return fstf



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fstf = main()
    plt.show()

if __name__=='__test__':
    fstf = main(test=True)
    np.testing.assert_equal(fstf['NY_Low'], 41)
    np.testing.assert_equal(fstf['WindTurbines'][0, 9], 50)
    try:
        os.remove(fstf.filename)
    except:
        pass

