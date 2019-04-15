from unittest import TestCase

from gator import cov

import numpy as np

class TestGenPropagate(TestCase):
    orig = np.diag(np.linspace(1., 3., 3))
    def test_requires_2d(self):
        with self.assertRaises(ValueError):
            cov_sub = cov.CovarianceSubselect(self.orig[0, :], 2)
            cov_sub = cov.CovarianceSubselect(self.orig[1:, :], 2)
            cov_sub = cov.CovarianceSubselect(self.orig[:, 1:], 2)
            cov_sub = cov.CovarianceSubselect(self.orig[:, :, None], 2)
            cov_sub = cov.CovarianceSubselect(self.orig, 10)

    def test_yields3d(self):
        cov_sub = cov.CovarianceSubselect(self.orig, 2)
        self.assertTrue(cov_sub.diag_windows.ndim == 3)

    def test_call_yieldscorrectshape(self):
        cov_sub = cov.CovarianceSubselect(self.orig, 2)
        self.assertTrue(cov_sub(1).ndim == 2)
        self.assertTrue(cov_sub(1).shape[0] == 2)