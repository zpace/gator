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

    def test_requires_correctshapevarmat(self):
        with self.assertRaises(gator.TransformationError):
            varmat_final = gator.propagate_varmat(self.varmat_orig[0, :], self.tmat)
            varmat_final = gator.propagate_varmat(self.varmat_orig[:, 0], self.tmat)
            varmat_final = gator.propagate_varmat(self.varmat_orig[0, 0], self.tmat)
            varmat_final = gator.propagate_varmat(self.varmat_orig[:-1, :], self.tmat)
            varmat_final = gator.propagate_varmat(self.varmat_orig[:, :-1], self.tmat)

    def test_requires_varmatliketmat(self):
        with self.assertRaises(gator.TransformationError):
            varmat_final = gator.propagate_varmat(self.varmat_orig[:-1, :-1], self.tmat)

class TestPropagateVarmats(TestPropagateVarmat):
    dim_orig = 3
    dim_final = 2
    start_arr = np.ones(2)
    tmat = np.ones((dim_orig, dim_final))
    varmat_orig = np.eye(dim_orig)
    broadcast_axis = (2, 3)
    n_within_broadcast = (3, 4)
    varmat_orig_large = np.tile(
        varmat_orig[..., None, None], (1, 1) + n_within_broadcast)

    def test_is_array(self):
        varmat_final = gator.propagate_varmats(
            self.varmat_orig_large, self.tmat, axis=self.broadcast_axis)
        self.assertTrue(isinstance(varmat_final, np.ndarray))

    def test_has_correct_dimension(self):
        varmat_final = gator.propagate_varmats(
            self.varmat_orig_large, self.tmat, axis=self.broadcast_axis)
        self.assertTrue(varmat_final.shape == (self.dim_final, self.dim_final) + self.n_within_broadcast)

    def test_requires_2dtmat(self):
        with self.assertRaises(gator.TransformationError):
            varmat_final = gator.propagate_varmats(
                self.varmat_orig_large, self.tmat[:, 0], axis=self.broadcast_axis)

    def test_requires_correctshapevarmat(self):
        with self.assertRaises(gator.TransformationError):
            varmat_final = gator.propagate_varmats(
                self.varmat_orig_large[:1, :, ...], self.tmat, axis=self.broadcast_axis)
            varmat_final = gator.propagate_varmats(
                self.varmat_orig_large[:, :1, ...], self.tmat, axis=self.broadcast_axis)
            varmat_final = gator.propagate_varmats(
                self.varmat_orig_large[:1, :1, ...], self.tmat, axis=self.broadcast_axis)
            varmat_final = gator.propagate_varmats(
                self.varmat_orig_large[:-1, :, ...], self.tmat, axis=self.broadcast_axis)
            varmat_final = gator.propagate_varmats(
                self.varmat_orig_large[:, :-1, ...], self.tmat, axis=self.broadcast_axis)

        def test_requires_varmatliketmat(self):
            with self.assertRaises(gator.TransformationError):
                varmat_final = gator.propagate_varmats(
                    self.varmat_orig_large[:-1, :-1, ...], self.tmat,
                    axis=self.broadcast_axis)

class TestGenPropagate(TestCase):
    dim_super = 10
    dim_orig = 4
    dim_final = 2
    K_super = np.eye(dim_super)  # larger covariance
    tmat = np.ones((dim_orig, dim_final))