import sys
import json
import os
from typing import Tuple, List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KernelDensity

from dagip.benchmark import *
from dagip.spatial import *
from dagip.retraction.probability_simplex import ProbabilitySimplex
from dagip.retraction.ratio import RatioManifold
from dagip.validation.k_fold import KFoldValidation
from dagip.retraction.base import Manifold


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'supervised-learning')
os.makedirs(RESULTS_FOLDER, exist_ok=True)


Xs = []

for MODALITY in ['long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions']:

    with open(os.path.join(DATA_FOLDER, 'D11-D12-batches.json'), 'r') as f:
        batch_dict = json.load(f)


    def load_cohort(folder: str) -> Tuple[np.ndarray, List[str]]:
        X = []
        batch_ids = []
        for filename in sorted(os.listdir(folder)):
            ega_id = os.path.splitext(filename)[0]
            if not filename.endswith('.csv'):
                continue
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath, sep=',')
            X.append(df.values[:, -1])
            batch_ids.append(batch_dict[ega_id])
        return np.asarray(X, dtype=float), batch_ids


    X_cases_D11, batch_cases_D11 = load_cohort(os.path.join(DATA_FOLDER, 'D11', 'cases', MODALITY))
    X_controls_D11, batch_controls_D11 = load_cohort(os.path.join(DATA_FOLDER, 'D11', 'controls', MODALITY))
    X_controls_D12, batch_controls_D12 = load_cohort(os.path.join(DATA_FOLDER, 'D12', 'controls', MODALITY))


    X = np.concatenate([X_cases_D11, X_controls_D11, X_controls_D12], axis=0)
    y = np.concatenate([np.ones(len(X_cases_D11), dtype=int), np.zeros(len(X_controls_D11) + len(X_controls_D12), dtype=int)])
    d = np.concatenate([np.zeros(len(X_cases_D11) + len(X_controls_D11), dtype=int), np.ones(len(X_controls_D12), dtype=int)])
    groups = LabelEncoder().fit_transform(batch_cases_D11 + batch_controls_D11 + batch_controls_D12)

    is_constant = np.all(X == X[0, np.newaxis, :], axis=0)
    X = X[:, ~is_constant]

    if MODALITY in {'fragment-length-distributions', 'end-motif-frequencies'}:

        if MODALITY == 'fragment-length-distributions':
            X = X[:, 40:]

        X = X / np.sum(X, axis=1)[:, np.newaxis]

    sample_ids = [str(i) for i in range(len(X))]
    Xs.append(X)


class MultimodalManifold(Manifold):

    def __init__(self, starts, ends):
        self.starts = starts
        self.ends = ends
        self.simplex_m = ProbabilitySimplex()
        self.ratio_m = RatioManifold()

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        flrp = self.ratio_m.transform(X[:, self.starts[0]:self.ends[0]])
        npsp = self.ratio_m.transform(X[:, self.starts[1]:self.ends[1]])
        emf = self.simplex_m.transform(X[:, self.starts[2]:self.ends[2]])
        fld = self.simplex_m.transform(X[:, self.starts[3]:self.ends[3]])
        return torch.cat([flrp, npsp, emf, fld], dim=1)

    def _inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        flrp = self.ratio_m.inverse_transform(X[:, self.starts[0]:self.ends[0]])
        npsp = self.ratio_m.inverse_transform(X[:, self.starts[1]:self.ends[1]])
        emf = self.simplex_m.inverse_transform(X[:, self.starts[2]:self.ends[2]])
        fld = self.simplex_m.inverse_transform(X[:, self.starts[3]:self.ends[3]])
        return torch.cat([flrp, npsp, emf, fld], dim=1)


sizes = np.cumsum([0] + [X.shape[1] for X in Xs])
starts = sizes[:-1]
ends = sizes[1:]
manifold = MultimodalManifold(starts, ends)
X = np.concatenate(Xs, axis=1)


METHODS = [
    (BaselineMethod(), 'baseline'),
    (CenteringScaling(), 'center-and-scale'),
    (KernelMeanMatching(), 'kmm'),
    (MappingTransport(), 'mapping-transport'),
    #(OTDomainAdaptation(manifold, SquaredEuclideanDistance(), 'tmp'), 'da'),
]

for method, method_name in METHODS:

    #if os.path.exists(os.path.join(RESULTS_FOLDER, f'{MODALITY}-{method_name}.json')):
    #    continue

    validation = KFoldValidation(
        X, y, d, sample_ids,
        target_domain=0, average_results=False, n_splits=5, n_repeats=30,
        groups=groups
    )

    results = {}
    results[method_name] = validation.validate(method)

    with open(os.path.join(RESULTS_FOLDER, f'BRCA-{method_name}.json'), 'w') as f:
        json.dump(results, f)
