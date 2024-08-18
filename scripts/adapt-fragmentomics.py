import sys
import json
import os
from typing import Tuple, List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dagip.core import DomainAdapter
from dagip.benchmark.baseline import BaselineMethod
from dagip.benchmark.centering_scaling import CenteringScaling
from dagip.benchmark.ot_da import OTDomainAdaptation
from dagip.spatial.euclidean import EuclideanDistance
from dagip.spatial.euclidean_log import EuclideanDistanceOnLog
from dagip.retraction.probability_simplex import ProbabilitySimplex
from dagip.retraction.ratio import RatioManifold
from dagip.validation.k_fold import KFoldValidation


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'corrected')
os.makedirs(RESULTS_FOLDER, exist_ok=True)


#for MODALITY in ['long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles']:
for MODALITY in ['end-motif-frequencies', 'fragment-length-distributions', 'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles']:


    with open(os.path.join(DATA_FOLDER, 'D11-D12-batches.json'), 'r') as f:
        batch_dict = json.load(f)


    def load_cohort(folder: str) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame]:
        X = []
        batch_ids, ega_ids = [], []
        for filename in os.listdir(folder):
            ega_id = os.path.splitext(filename)[0]
            if not filename.endswith('.csv'):
                continue
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath, sep=',')
            X.append(df.values[:, -1])
            batch_ids.append(batch_dict[ega_id])
            ega_ids.append(ega_id)
        return np.asarray(X, dtype=float), batch_ids, ega_ids, df


    X_cases_D11, batch_cases_D11, _, _ = load_cohort(os.path.join(DATA_FOLDER, 'D11', 'cases', MODALITY))
    X_controls_D11, batch_controls_D11, _, _ = load_cohort(os.path.join(DATA_FOLDER, 'D11', 'controls', MODALITY))
    X_controls_D12, batch_controls_D12, ega_ids_D12, sample_df = load_cohort(os.path.join(DATA_FOLDER, 'D12', 'controls', MODALITY))


    X = np.concatenate([X_cases_D11, X_controls_D11, X_controls_D12], axis=0)
    y = np.concatenate([np.ones(len(X_cases_D11), dtype=int), np.zeros(len(X_controls_D11) + len(X_controls_D12), dtype=int)])
    d = np.concatenate([np.zeros(len(X_cases_D11) + len(X_controls_D11), dtype=int), np.ones(len(X_controls_D12), dtype=int)])
    groups = LabelEncoder().fit_transform(batch_cases_D11 + batch_controls_D11 + batch_controls_D12)

    if MODALITY in {'fragment-length-distributions', 'end-motif-frequencies'}:

        #if MODALITY == 'fragment-length-distributions':
        #    X = X[:, 40:]

        X = X / np.sum(X, axis=1)[:, np.newaxis]
        X_controls_D11 = X_controls_D11 / np.sum(X_controls_D11, axis=1)[:, np.newaxis]
        X_controls_D12 = X_controls_D12 / np.sum(X_controls_D12, axis=1)[:, np.newaxis]

    if MODALITY in {'fragment-length-distributions', 'end-motif-frequencies'}:
        manifold = ProbabilitySimplex()
        pairwise_distances = EuclideanDistanceOnLog()
    else:
        manifold = RatioManifold()
        pairwise_distances = EuclideanDistance()

    model = DomainAdapter(manifold=manifold)
    X_controls_D12_adapted = model.fit_transform(X_controls_D12, X_controls_D11)

    for i in range(len(X_controls_D12_adapted)):
        os.makedirs(os.path.join(RESULTS_FOLDER, 'D12', 'controls', MODALITY), exist_ok=True)
        filepath = os.path.join(RESULTS_FOLDER, 'D12', 'controls', MODALITY, f'{ega_ids_D12[i]}.csv')
        if MODALITY in {'fragment-length-distributions', 'end-motif-frequencies'}:
            sample_df['Count'] = X_controls_D12_adapted[i, :]
        elif MODALITY == 'long-fragment-ratio-profiles':
            sample_df['Ratio'] = X_controls_D12_adapted[i, :]
        else:
            sample_df['Average score'] = X_controls_D12_adapted[i, :]
        sample_df.to_csv(filepath, index=False)

