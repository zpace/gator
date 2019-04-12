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
