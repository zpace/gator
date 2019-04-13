import numpy as np

class TransformationError(Exception):
    '''Exception denoting an error in linear transformation
    '''
    pass

# propagate variance matrix
def propagate_varmat(varmat, tmat):
    '''Propagate variance-covariance matrix through linear transformation
    
    Executes pre- and post-multiplication of a variance-covariance 
    matrix by a transformation matrix
    
    Parameters
    ----------
    varmat : array-like
        2D variance-covariance matrix with `shape` `(n, n)`
    tmat : array-like
        2D transformation matrix with `shape` `(n, p)`, which transforms 
        a vector of `len` `n` to a vector of `len` `p`

    Returns
    -------
    new_varmat : array-like
        2D variance-covariance matrix with `shape` `(p, p)`
    '''

    if varmat.ndim != 2:
        raise TransformationError('variance matrix must be 2D')

    if varmat.shape[0] != varmat.shape[1]:
        raise TransformationError('variance matrix must be square')

    if tmat.ndim != 2:
        raise TransformationError('transformation matrix must be 2D')

    if tmat.shape[0] != varmat.shape[0]:
        raise TransformationError('zeroth axis of transformation matrix must match zeroth axis of variance matrix')

    new_varmat = np.linalg.multi_dot([tmat.T, varmat, tmat])

    return new_varmat

def propagate_varmats(varmats, tmat, axis=-1):
    '''Propagate a series of variance matrices through a single linear transformation
    
    Executes vectorized pre- and post-multiplication of a variance-covariance
    matrix by a transformation matrix
    
    Parameters
    ----------
    varmats : array-like
        series of 2D variance-covariance matrix with `shape` `(n, n)`,
        all stacked along one or more axes
    tmat : array-like
        2D transformation matrix with `shape` `(n, p)`, which transforms 
        a vector of `len` `n` to a vector of `len` `p`
    axis : {int, sequence}, optional
        the axis (or axes) along with the matrix-multiplication is
        broadcast (the default is -1, which means the variance matrices
        are stacked along the final axis)
    '''

    if type(axis) is int:
        axis = tuple(axis)
    elif type(axis) in (tuple, list):
        pass
    else:
        raise ValueError('axis must be integer or list/tuple')

    if varmats.ndim <= 2:
        raise TransformationError('variance matrix must be >2D')

    if tmat.ndim != 2:
        raise TransformationError('transformation matrix must be 2D')

    num_broadcast_axes = len(axis)

    # first, rearrange axes of varmats, s.t. broadcast axes end up first
    varmats = np.moveaxis(varmats, axis, range(num_broadcast_axes))

    if varmats.shape[num_broadcast_axes:][0] != varmats.shape[num_broadcast_axes:][1]:
        raise TransformationError('variance matrix must be square')

    if tmat.shape[0] != varmats.shape[:num_broadcast_axes][0]:
        raise TransformationError('zeroth axis of transformation matrix must match zeroth axis of variance matrix')

    new_varmats = np.einsum('ia,...ij,jb->...ab', tmat, varmats, tmat)

    # rearrange result to have axes in the same order as the matrix
    # that was originally passed
    new_varmats = np.moveaxis(new_varmats, range(num_broadcast_axes), axis)

    return new_varmats
    
