Welcome to gator's documentation!
=================================

gator (pronounced "gā-tər") is a python package which propagates observational uncertainties through linear-algebraic transformations

Installation
============

`gator` can be `pip` installed!

Example Usage
=============

.. code-block:: python3

    import numpy as np
    from sklearn.datasets import make_spd_matrix

    n_orig = 10
    n_final = 4

    obs = np.random.randn(n_orig)
    print(obs)

    obs_covar = make_spd_matrix(n_orig)
    new_param_space_covar = make_spd_matrix(n_orig)
    evals, evecs = np.linalg.eigh(new_param_space_covar)
    tfm = evecs[:, ::-1][:, :n_final]
    obs_tfm = obs @ tfm
    obs_covar_tfm = propagate_varmat(obs_covar, tfm)

    print(obs_covar_tfm)

`gator` includes tools for propagating uncertainties through linear transformations, and dealing with selecting from high-dimesional covariance matrices

Linear Transformations
======================

.. automodule:: gator.gator
    :members:

Covariances
===========

.. automodule:: gator.cov
    :members:


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
