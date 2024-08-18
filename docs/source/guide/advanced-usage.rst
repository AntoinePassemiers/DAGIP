Advanced usage
==============

Custom manifold
---------------

To correct each sample :math:`X_{i\cdot}` in the source domain, the adapter uses the following function:

.. math::

    \mathcal{X} = f(f^{-1}(X) + g(X))

where ``f`` is a differentiable function that maps each row of an input matrix onto a manifold, and ``g`` is a multi-layer perceptron taking p-dimensional vectors as input and producing vectors of the same size. Intuitively, ``g`` is a model used to explicitly learn the bias between the source and the target domains.

To implement a new manifold, both :math:`f` and :math:`f^{-1}` should be defined. :math:`f` and :math:`f^{-1}` correspond to the ``_transform`` and ``_inverse_transform`` abstract methods from the ``dagip.retraction.base.Manifold`` base class. 

Let's take as example the multinomial manifold, namely the manifold of matrices with positive elements and having their rows summing up to 1 each:

.. code-block:: python

    import torch
    from dagip.retraction.base import Manifold


    class ProbabilitySimplex(Manifold):

        def __init__(self, eps: float = 1e-6):
            # Constant used to prevent numerical issues
            self.eps: float = eps

        def _transform(self, X: torch.Tensor) -> torch.Tensor:
            # Project X onto the multinomial manifold
            return torch.softmax(X, dim=1)

        def _inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
            X = torch.clamp(X, self.eps, 1)  # Avoid numerical issues
            return torch.log(X)  # Project X back in the Euclidean space

``_transform`` projects the data from the Euclidean space to the given manifold, while ``_inverse_transform`` performs the reverse mapping, from the manifold to the Euclidean space. Let's note that in the given example, the logarithm is indeed the inverse of the softmax operation, since ``f^{-1}`` is called before ``f``, and ``X`` is assumed to have its rows summing to 1 beforehand. The assumption that ``X`` is already on the manifold can be exploited to easily implement otherwise non-invertible functions.

Multimodal analysis
-------------------

In case multiple modalities are needed to be corrected jointly, constraints on the data can be defined using a ``MultimodalManifold``. For example, 4-mer end motif frequencies (256 possible motifs) and methylation ratios from 1500 differentially methylated regions (DMRs) can be constrained using the following manifold:

.. code-block:: python

    from dagip.retraction import *

    manifold = MultimodalManifold()
    manifold.add(256, ProbabilitySimplex())
    manifold.add(1500, RatioManifold())

Since the inputs ``X`` and ``Y`` to the algorithm should be NumPy arrays, data from all modalities should be concatenated into single arrays. In the given examples, the first 256 columns of ``X`` and ``Y`` correspond to end motif frequencies, while the rest correspond to methylation ratios.

Custom distance metric
----------------------

Implementing custom metrics is more straightforward, as only the ``pairwise_distances`` method needs to be defined. Because ``X`` and ``Y`` are of dimensions `(n, p)` and `(m, p)`, respectively, the output of the method should be a PyTorch tensor of shape `(n, m)`. The method should be differentiable.

.. code-block:: python

    import torch
    from dagip.spatial.base import BaseDistance


    class ManhattanDistance(BaseDistance):

        def pairwise_distances(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            return torch.cdist(X, Y, p=1)
