import unittest
import numpy as np
import os
MyDir=os.path.dirname(__file__)
from openfast_toolbox.airfoils.Polar import * 

# --------------------------------------------------------------------------------}
# ---  
# --------------------------------------------------------------------------------{
class TestPolarManip(unittest.TestCase):
    def assertNaN(self,x):
        self.assertTrue(np.isnan(x))

    def test_read(self):
        P=Polar(os.path.join(MyDir,'../data/FFA-W3-241-Re12M.dat'))
        self.assertEqual(P.alpha[-1],180)
        self.assertEqual(P.cl[-1],0)

        P=Polar(os.path.join(MyDir,'../data/Cylinder.dat'))
        self.assertEqual(P.cl.size,3)

    def test_correction3d_angle_units(self):
        degrees = Polar(os.path.join(MyDir, '../data/FFA-W3-241-Re12M.dat'))
        radians = Polar(Re=degrees.Re, alpha=np.deg2rad(degrees.alpha),
                        cl=degrees.cl, cd=degrees.cd, cm=degrees.cm, radians=True)
        limits = dict(alpha_max_corr=12, alpha_linear_min=-4.1, alpha_linear_max=4.1)

        for options in ({}, {'blending_method': 'heaviside'},
                        dict(limits, blending_method='heaviside')):
            with self.subTest(options=options):
                expected = degrees.correction3D(0.197, 0.232, 9.153, drag_method='Eggers', **options)
                actual = radians.correction3D(0.197, 0.232, 9.153, drag_method='Eggers', **options)
                self.assertFalse(np.allclose(expected.cl, degrees.cl))
                np.testing.assert_allclose(actual.alpha, expected.alpha, atol=1e-12)
                np.testing.assert_allclose(actual.cl, expected.cl, atol=1e-12)
                np.testing.assert_allclose(actual.cd, expected.cd, atol=1e-12)

if __name__ == '__main__':
    unittest.main()
