import os
import unittest
from pathlib import Path
from openfast_toolbox.fastfarm.FASTFarmCaseCreation import FFCaseCreation

class TestMoorDynSupport(unittest.TestCase):
    def setUp(self):
        """
        Setup the testing environment.
        """
        # Create a temporary directory for the test
        self.test_dir = Path("test_moordyn_support")
        self.test_dir.mkdir(exist_ok=True)

        # Define MoorDyn template
        self.moordyn_template = self.test_dir / "MoorDyn_template.dat"
        self.moordyn_template.write_text(
            """Node    X     Y     Z    M     B
            0.0   0.0  -20.0  0.0   0.0
            100.0 0.0  -20.0  0.0   0.0
            0.0   100.0 -20.0 0.0   0.0
            """
        )
        # Initialize FFCaseCreation with minimal parameters
        self.case = FFCaseCreation(
            path=str(self.test_dir),
            wts={
                0: {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'D': 240,          # Rotor diameter
                    'zhub': 150,       # Hub height
                    'cmax': 5,         # Maximum blade chord (m)
                    'fmax': 10 / 6,    # Maximum excitation frequency (Hz)
                    'Cmeander': 1.9    # Meandering constant (-)
                }
            },
            tmax=600,
            zbot=1.0,
            vhub=[10.0],
            shear=[0.2],
            TIvalue=[10],
            inflow_deg=[30.0],  # Rotate MoorDyn file by 30 degrees
            dt_high_les=0.6,
            ds_high_les=10.0,
            extent_high=1.2,
            dt_low_les=3.0,
            ds_low_les=20.0,
            extent_low=[3, 8, 3, 3, 2],
            ffbin=None,
            mod_wake=1,
            yaw_init=None,
            nSeeds=1,
            LESpath=None,
            refTurb_rot=0,
            verbose=1,
        )

    def tearDown(self):
        """
        Cleanup after tests.
        """
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()

    def test_moordyn_file_copy_and_rotation(self):
        """
        Test the copying and rotation of the MoorDyn file.
        """
        case = self.case
        # Set the MoorDyn template
        case.setTemplateFilename(str(self.test_dir), {"mDynfilename": self.moordyn_template.name})

        # Simulate case generation
        case.copyTurbineFilesForEachCase()

        # Verify MoorDyn file is created
        output_file = self.test_dir / "case_0_inflow30_Seed0" / "MoorDyn.dat"
        self.assertTrue(output_file.exists(), "MoorDyn file was not created")

        # Check the MoorDyn file content for rotation
        with open(output_file, "r") as f_out:
            rotated_lines = f_out.readlines()

        # Expected rotated values (30 degrees rotation)
        import numpy as np
        rotation_matrix = np.array([
            [np.cos(np.radians(30)), -np.sin(np.radians(30)), 0],
            [np.sin(np.radians(30)), np.cos(np.radians(30)), 0],
            [0, 0, 1],
        ])
        expected_coordinates = [
            [0.0, 0.0, -20.0],
            [100.0, 0.0, -20.0],
            [0.0, 100.0, -20.0],
        ]
        rotated_coordinates = [np.dot(rotation_matrix, np.array(coord)) for coord in expected_coordinates]

        # Validate each node's position
        for i, expected_coord in enumerate(rotated_coordinates):
            parts = rotated_lines[i + 1].split()
            x, y, z = map(float, parts[1:4])
            self.assertAlmostEqual(x, expected_coord[0], places=4, msg=f"Node {i} X mismatch")
            self.assertAlmostEqual(y, expected_coord[1], places=4, msg=f"Node {i} Y mismatch")
            self.assertAlmostEqual(z, expected_coord[2], places=4, msg=f"Node {i} Z mismatch")


if __name__ == "__main__":
    unittest.main()
