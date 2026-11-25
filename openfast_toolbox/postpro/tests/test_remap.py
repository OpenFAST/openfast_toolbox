import unittest

import numpy as np
import pandas as pd

from openfast_toolbox.postpro import postpro
from openfast_toolbox.tools import pandalib


class TestRemap(unittest.TestCase):

    def _check_remap(self, remap_df):
        df = pd.DataFrame({'Wind1VelX_[m/s]': [8.0, 9.0], 'RotSpeed_[rpm]': [10.0, 12.0]})
        ColMap = {
            'WS_[m/s]'         : 'Wind1VelX_[m/s]',               # rename
            'RotSpeed_[rad/s]' : '{RotSpeed_[rpm]} * 2*np.pi/60', # new column
        }
        df2 = remap_df(df, ColMap)
        self.assertEqual(list(df2.columns), ['WS_[m/s]', 'RotSpeed_[rpm]', 'RotSpeed_[rad/s]'])
        np.testing.assert_allclose(df2['WS_[m/s]'], [8.0, 9.0])
        np.testing.assert_allclose(df2['RotSpeed_[rad/s]'], np.array([10.0, 12.0])*2*np.pi/60)
        # The input dataframe is not modified
        self.assertEqual(list(df.columns), ['Wind1VelX_[m/s]', 'RotSpeed_[rpm]'])

    def test_remap_postpro(self):
        self._check_remap(postpro.remap_df)

    def test_remap_pandalib(self):
        self._check_remap(pandalib.remap_df)


if __name__ == '__main__':
    unittest.main()
