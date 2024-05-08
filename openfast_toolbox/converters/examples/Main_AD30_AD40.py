""" 
Converts a AD3.x file to AD4.0
"""
import os
import numpy as np
from openfast_toolbox.converters.versions.aerodyn import ad30_to_ad40

# Get current directory so this script can be called from any location
scriptDir=os.path.dirname(__file__)

fileold = os.path.join(scriptDir, '../../../data/example_files/AeroDyn_v3.0.dat')
filenew = '_aerodyn_temp.dat'

if __name__ == '__main__':
    ad = ad30_to_ad40(fileold, filenew, verbose=True, overwrite=False)

if __name__ == '__test__':
    ad = ad30_to_ad40(fileold, filenew, verbose=False, overwrite=False)
    # small tests
    np.testing.assert_equal(ad['Wake_Mod'], 1)
    np.testing.assert_equal(ad['UA_Mod'], 3)
    np.testing.assert_equal(ad['DBEMT_Mod'], 2)
    os.remove(filenew)
