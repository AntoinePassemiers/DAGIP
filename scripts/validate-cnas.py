import sys
import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dagip.benchmark.no_target import NoTarget
from dagip.benchmark.no_source import NoSource
from dagip.benchmark import *
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.spatial import *
from dagip.retraction.positive import Positive
from dagip.retraction.gip import GIPManifold
from dagip.validation.k_fold import KFoldValidation


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'supervised-learning')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'HL', 'DLBCL', 'MM', 'OV',
    ],
    help='Dataset name'
)
args = parser.parse_args()
DATASET = args.dataset

# Load reference GC content
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc-content-1000kb.csv'))
gc_content = df['MEAN'].to_numpy()
bin_chr_names = df['CHR'].to_numpy()
bin_starts = df['START'].to_numpy()
bin_ends = df['END'].to_numpy()

# Load data
filename = 'OV.npz' if (DATASET == 'OV') else 'HEMA.npz'
data = np.load(os.path.join(ROOT, DATA_FOLDER, 'numpy', filename), allow_pickle=True)
gc_codes = data['gc_codes']
gc_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
X = data['X']
y = data['y']
d = np.squeeze(LabelEncoder().fit_transform(data['d'][:, np.newaxis]))
t = data['t']
groups = np.squeeze(LabelEncoder().fit_transform(data['groups'][:, np.newaxis]))

# Define labels
if DATASET == 'OV':
    y = (y == 'OV').astype(int)
elif DATASET == 'HEMA':
    y = (y != 'Healthy').astype(int)
else:
    mask = np.asarray([label in {'Healthy', DATASET} for label in y], dtype=bool)
    X, y, d, t, groups, gc_codes = X[mask], y[mask], d[mask], t[mask], groups[mask], gc_codes[mask]
    y = (y == DATASET).astype(int)
assert np.all(d < 2)

# Ensure there is at least one cancer case in the reference domain.
# Otherwise, swap domains.
if not np.any(y[d == 0]):
    d = 1 - d

# Remove problematic regions
mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
X = X[:, mask]
gc_content = gc_content[mask]
bin_chr_names = bin_chr_names[mask]
bin_starts = bin_starts[mask]
bin_ends =  bin_ends[mask]

# GC-correction
X = gc_correction(X, gc_content)

print('Start validation')

manifold = Positive()
pairwise_distances = SquaredEuclideanDistance()

METHODS = [
    (BaselineMethod(), 'baseline'),
    #(CenteringScaling(), 'center-and-scale'),
    #(KernelMeanMatching(), 'kmm'),
    #(MappingTransport(per_label=False), 'mapping-transport'),  # per_label=True makes the model crash for some reason
    #(OTDomainAdaptation(manifold, pairwise_distances, 'tmp'), 'da'),
    #(DryClean(bin_chr_names, bin_starts, bin_ends, f'tmp/dryclean/{DATASET}'), 'dryclean'),
]

for method, method_name in METHODS:

    n_repeats = 30 if (method_name != 'dryclean') else 3

    if os.path.exists(os.path.join(RESULTS_FOLDER, f'{DATASET}-{method_name}.json')):
        continue

    print(f'Processing {DATASET}-{method_name}')

    # Cross-validation
    validation = KFoldValidation(
        X, y, d, gc_codes,
        target_domain=0, average_results=False, n_splits=5, n_repeats=n_repeats,
        groups=groups,
    )
    results = {}
    results[method_name] = validation.validate(method)

    with open(os.path.join(RESULTS_FOLDER, f'{DATASET}-{method_name}.json'), 'w') as f:
        json.dump(results, f)
