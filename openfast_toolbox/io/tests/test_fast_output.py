import unittest
import os
import numpy as np
from openfast_toolbox.io.tests.helpers_for_test import MyDir, reading_test 
from openfast_toolbox.io import FASTOutputFile

class Test(unittest.TestCase):

    def test_001_read_all(self, DEBUG=True):
        reading_test('FASTOut*.*', FASTOutputFile)

    def DF(self,FN):
        """ Reads a file and return a dataframe """ 
        return FASTOutputFile(os.path.join(MyDir,FN)).toDataFrame()
 
    def test_FASTOut(self):
        self.assertEqual(self.DF('FASTOut.out').values[-1,1],1036)
 
    def test_FASTOutBin(self):
        # --- Test reading
        F = FASTOutputFile(os.path.join(MyDir,'FASTOutBin.outb'))
        M = F.toDataFrame()
        self.assertAlmostEqual(M['GenPwr_[kW]'].values[-1],40.57663190807828)
        # --- Test writing
        tempFilename = '_FASTOutBin_out.outb'
        # Write to tempfile
        F.write(tempFilename)
        # Read written file
        F2= FASTOutputFile(tempFilename)
        # Test that read data match
        np.testing.assert_almost_equal(F.data.values,F2.data.values, 4)
        np.testing.assert_almost_equal(F.data.values[-1,-1] ,40.57663190807828, 10)
        np.testing.assert_almost_equal(F2.data.values[-1,-1],40.57663190807828, 10)
        self.assertEqual(F2.channels[-1],'GenPwr')
        self.assertEqual(F2.units[-1],'kW')
        # cleanup
        try:
            os.remove(tempFilename)
        except:
            pass

    def test_FASTOutBin_uncompressed(self):
        # Uncompressed binary (FileID 3), written by OpenFAST with OutFileFmt=4
        filename = os.path.join(MyDir,'FASTOutBin_ID3.outb')
        M = FASTOutputFile(filename).toDataFrame()
        np.testing.assert_allclose(M['Time_[s]'].values, [0, 0.025, 0.05, 0.075, 0.1])
        # Values from the text output of the same run
        np.testing.assert_allclose(M['TwrBsFxt_[kN]'].values, [10.448863, 0.17119217, 8.0563698, -1.2610151, 26.274128], rtol=1e-7)
        # Buffered reading gives the same data
        M2 = FASTOutputFile(filename, use_buffer=True).toDataFrame()
        np.testing.assert_array_equal(M.values, M2.values)

if __name__ == '__main__':
#     Test().test_000_debug()
    unittest.main()
