Quick start
===========

Installation
------------

The package can be installed with:

.. code-block:: bash
    
    pip install -r requirements.txt
    python setup.py install

Usage
-----

In order to use ``dagip``, two cohorts representative of the same population should be available. Typically, each cohort should be at least of size 40. More importantly, these cohorts should share similar biological characteristics (e.g., controls matched for age and sex). For each cohort, the data should be structured as a two-dimensional NumPy array. For the target domain (the domain where final downstream analysis will take place), the matrix is denoted by ``Y``, a NumPy array of size `(m, q)`, where each row corresponds to a sample. For the source domain (the "biased" domain, containing the samples to be corrected), the matrix is denoted by ``X``, a NumPy array of size `(n, q)`, where each row corresponds to a sample.

In the general case, bias correction can be performed in the following way:

.. code-block:: python

    from dagip.core import DomainAdapter

    model = DomainAdapter()
    X_adapted = model.fit_transform(X, Y)

The output ``X_adapted`` is the corrected version of ``X``. Correction is performed in such a way that ``X_adapted`` matches the distribution of ``Y``.

Alternatively, the samples can be split into groups. For example, if male and female control cohorts are available, and you require males to be mapped to males exclusively (and vice versa), then lists of NumPy arrays can be provided instead. Another example is grouping the controls and cancer cases separately. The algorithm is called in the exact same way:

.. code-block:: python

    X_adapted = model.fit_transform(Xs, Ys)

where ``Xs`` and ``Ys`` are lists of the same length. Each element is a NumPy array of arbitrary number of rows, and number of columns equal to ``q``.

After inference, the trained model can be used to adapt any new sample in the source domain independently, without having recourse to any additional target group:

.. code-block:: python

    X_new_adapted = model.transform(X_new)

where ``X_new`` contains new samples to be corrected.

Adding constraints
------------------

By default, the algorithm does not guarantee that the corrected data lies on the same manifold as the original data. For example, if ``X`` consists of methylation ratios (i.e., beta values), then each value should lie in the ``[0, 1]`` range. To add this constraint to the model, you can specify it through the ``manifold`` argument:

.. code-block:: python

    from dagip.core import DomainAdapter
    from dagip.retraction import RatioManifold

    model = DomainAdapter(manifold=RatioManifold())

If your data contains a lot of zeros, please use this feature carefully (due to numerical instability).

Currently, the following manifolds are available:

:Identity: Default manifold. Does not apply any constraint.
:Positive: Enforces the positivity of each element of the matrix.
:ProbabilitySimplex: Enforces the positivity of each element of the matrix, as well as each of its row summing up to 1.
:RatioManifold: Enforces each element of the matrix to stay in the ``[0, 1]`` range.
:GIPManifold: Enforces the positivity before performing GC-correction on each row of the matrix.

Coverage profiles
-----------------

To run the algorithm on coverage profiles, consider using the ``Positive`` or even the ``GIPManifold``. This manifold requires that the rows in ``X`` and ``Y`` have been GC-corrected already. To avoid introducing any discrepancy, prior GC-correction should be identical to the GC-correction performed by ``GIPManifold``: algorithm and hyper-parameters should be the same. To ensure this, consider using our implementation:

.. code-block:: python

    from dagip.correction.gc import gc_correction

    X = gc_correction(X, gc_content, frac=0.3)
    Y = gc_correction(Y, gc_content, frac=0.3)

where ``frac`` is the fraction of data points used to build each local model (that is, the level of smoothing), and gc_content is a NumPy array of size ``(q,)`` containing the GC-content ratio of each bin.

Then, the manifold should be specified:

.. code-block:: python

    from dagip.core import DomainAdapter
    from dagip.retraction import GIPManifold

    model = DomainAdapter(manifold=GIPManifold(gc_content, frac=0.3))

Let's note that using the ``Positive`` manifold may be sufficient, as prior GC-correction can imply that the corrected coverage profiles remain decorrelated from GC content. If that is the case, consider using the ``Positive`` manifold to avoid additional computations and potential over-correction.

Saving and loading the adapter
------------------------------

.. code-block:: python

    # Saving
    model.save('/some/location.pt')

    # Loading
    model.load('/some/location.pt')

Hyper-parameters
----------------

The algorithm has other hyper-parameters:

.. code-block:: python

    from dagip.core import DomainAdapter, default_u_test
    from dagip.retraction import Identity
    from dagip.spatial.euclidean import EuclideanDistance

    model = DomainAdapter(
        folder='/somewhere',
        manifold=Identity(),
        pairwise_distances=EuclideanDistance(),
        u_test=default_u_test,
        var_penalty=0.01,
        reg_rate=0.1,
        max_n_iter=4000,
        convergence_threshold=0.5,
        nn_n_hidden=32,
        nn_n_layers=4,
        lr=0.005,
        verbose=True
    )

Hyper-parameter list:

:folder: Folder where to store figures.
:manifold: ``dagip.retraction.base.Manifold`` instance. Manifold used to add constraints on the data matrix. Please refer to :doc:`this section <advanced-usage>` for implementing custom manifolds.
:distance: ``dagip.spatial.base.BaseDistance`` instance. Used to define the cost matrix and solve the optimal transport problem. Please refer to :doc:`this section <advanced-usage>` for implementing custom distance metrics. Default: ``SquaredEuclideanDistance``.
:u_test: Univariate statistical test which will be performed on each variable separately. Should be a function taking two arguments, ``x`` and ``y`` (both 1-dimensional NumPy arrays), and returns a p-value. Default: Two-sample Kolmogorov-Smirnov test.
:reg_rate: Initial value of the regularization rate. A large value reduces the chances to introduce large changes in the data. If the two cohorts ``X`` and ``Y`` are expected to be perfectly superimposed after correction (for example if ``Y`` contains technical replicates of samples in ``X``), then ``reg_rate`` can be set to a low value instead.
:u_loss_weight: Importance of the univariate Wasserstein distances in the total loss function. Increasing this value encourages the matching between medians and inter-quartile ranges.
:max_n_iter: Maximum number of iterations of the algorithm.
:convergence_threshold: Cutoff on the median of p-values computed with the ``u_test`` function. When the median p-value exceeds that threshold, the algorithm stops.
:nn_n_hidden: Number of hidden neurons in each layer of the neural network model.
:nn_n_layers: Number of layers in the neural network model.
:lr: Learning rate used to update the parameters of the neural network model.
:l2_reg: L2 regularization of the neural network parameters.
:batch_size: Number of samples used to perform a forward pass during training of the neural netwwork. Can affect convergence speed and computation times.
:verbose: Whether to print debugging information.
