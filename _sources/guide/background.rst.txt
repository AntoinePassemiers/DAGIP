Background
==========

.. math::

	W_p(C) = \left(\sum_{i=1}^n\sum_{j=1}^m C_{ij}^p \Gamma'_{ij}\right)^{1/p}

where :math:`\Gamma'` is the optimal transport plan found by solving the following linear problem:

.. math::
	:nowrap:

	\begin{equation}
	\begin{aligned}
		\text{min}_{\ \Gamma} \quad & \sum_{i=1}^n\sum_{j=1}^m C_{ij}^p \Gamma_{ij} \\
		\text{s.t.} \quad & \sum_{j=1}^m \Gamma_{ij} = \nu_i & \quad \forall i \\
		\quad & \sum_{i=1}^n \Gamma_{ij} = \mu_j & \quad \forall j \\
		\quad & \Gamma_{ij} \ge 0 & \quad \forall i, j \\
	\end{aligned}
	\end{equation}

.. math::
	:nowrap:

	\begin{equation}
	\begin{aligned}
	\min_{\Theta} \quad & \left( \min_{\Gamma \in \mathcal{F}} \ \sum_{i=1}^n\sum_{j=1}^m C_{ij}^p \Gamma_{ij} \right)^{1/p} \\
	\text{where} \quad & C = d\left( f_{\Theta}(E), Y \right)
	\label{eq:prob}
	\end{aligned}
	\end{equation}
