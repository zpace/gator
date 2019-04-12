from unittest import TestCase

import gator

import numpy as np

class TestPropagateVarmat(TestCase):
    dim_orig = 3
    dim_final = 2
    start_arr = np.ones(2)
    tmat = np.ones((dim_orig, dim_final))
    varmat_orig = np.eye(dim_orig)
    def test_is_array(self):
        varmat_final = gator.propagate_varmat(self.varmat_orig, self.tmat)
        self.assertTrue(isinstance(varmat_final, np.ndarray))

    def test_has_correct_dimension(self):
        varmat_final = gator.propagate_varmat(self.varmat_orig, self.tmat)
        self.assertTrue(varmat_final.shape == (self.dim_final, self.dim_final))

    def test_requires_2dtmat(self):
        with self.assertRaises(gator.TransformationError):
            varmat_final = gator.propagate_varmat(self.varmat_orig, self.tmat[:, 0])

    def test_requires_2dsquarevarmat(self):
        with self.assertRaises(gator.TransformationError):
            varmat_final = gator.propagate_varmat(self.varmat_orig[0, :], self.tmat)
            varmat_final = gator.propagate_varmat(self.varmat_orig[:, 0], self.tmat)
            varmat_final = gator.propagate_varmat(self.varmat_orig[0, 0], self.tmat)
            varmat_final = gator.propagate_varmat(self.varmat_orig[:-1, :], self.tmat)
            varmat_final = gator.propagate_varmat(self.varmat_orig[:, :-1], self.tmat)

    def test_requires_varmatliketmat(self):
        with self.assertRaises(gator.TransformationError):
            varmat_final = gator.propagate_varmat(self.varmat_orig[:-1, :-1], self.tmat)