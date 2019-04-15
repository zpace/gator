from astropy.utils.decorators import lazyproperty

import numpy as np
from numpy.lib.stride_tricks import as_strided

class CovarianceSubselect(object):
    '''select fixed-size intervals of a covariance matrix

    Parameters
    ----------
    cov : array-like
        full-sized covariance matrix
    n : int
        size of intervals to be selected

    Attributes
    ----------
    cov : ndarray
        full-sized covariance matrix
    n : int
        size of intervals to be selected

    Examples
    --------
    >>> covbase = np.diag(np.linspace(1., 3., 3))
    >>> cov_sub = CovarianceSubselect(covbase, 2)
    >>> cov_sub(0)
    array([[1., 0.]
           [0., 2.]])
    >>> cov_sub[1]
    array([[2., 0.]
           [0., 3.]])

    '''
    def __init__(self, cov, n):
        self.cov = cov
        self.n = n
        _ = self.diag_windows  # evaluate this once so that input is checked

    @lazyproperty
    def diag_windows(self):
        '''strided view on `self.cov`
        
        Returns
        -------
        ndarray
            view of `self.cov`
        '''
        return diag_windows(self.cov, self.n)

    def __call__(self, start):
        '''retrieve window of `self.cov` with some starting index
        
        Parameters
        ----------
        start : int
            starting index

        Returns
        -------
        ndarray
            view of `self.cov`
        '''
        return self.diag_windows[start]

    def __getitem___(self, *args, **kwargs):
        return self.diag_windows.__getitem__(*args, **kwargs)

def diag_windows(x, n):
    '''strided view on a 2D input array, centered on the main diagonal
    
    use `as_strided` to return a series of views on an input array,
    where each element is a sub-array centered on the main diagonal,
    with a specified size
    
    Parameters
    ----------
    x : ndarray
        large array, for which views are generated
    n : int
        size of a view along each axis
    
    Returns
    -------
    ndarray
        a 3D array composed of views of `x`, each with `shape` `(n, n)`
    
    Raises
    ------
    ValueError
        x must be 2D, square, and larger than view size


    Examples
    --------
    >>> x = np.diag(np.linspace(1., 3., 3))
    >>> xview = diag_windows(x, 2)
    array([[[1., 0.],
            [0., 2.]],
           [[2., 0.],
            [0., 3.]]])
    '''
    if x.ndim != 2:
        raise ValueError('x must be 2D')
    if (x.shape[0] != x.shape[1]):
        raise ValueError('x must be square')
    if x.shape[0] < n:
        raise ValueError('size of view must be smaller than size of x')
    w = as_strided(x, shape=(x.shape[0] - n + 1, n, n),
                   strides=(x.strides[0]+x.strides[1], x.strides[0], x.strides[1]))
    return w