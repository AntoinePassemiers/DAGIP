import os
import json

import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'pairs-raw')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(FIGURES_FOLDER, exist_ok=True)

SIGNIFICANCE = False

DATASETS = ['OV-forward', 'NIPT-chemistry', 'NIPT-lib', 'NIPT-adapter', 'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000']
METRICS = ['accuracy', 'r2', 'mcnemar-pvalue', 'ztest-pvalue']
if SIGNIFICANCE:
    METRIC_NAMES = [r"McNemar's test $p$-values", r'$R^2$ $z$-test $p$-values']
else:
    METRIC_NAMES = ['Pairing accuracy based on Euclidean distance', r'Coefficient of determination $R^2$']
METHODS = ['baseline', 'centering-scaling', 'mapping-transport', 'dryclean', 'da']
METHOD_NAMES = ['Baseline', 'Center-and-scale', 'MappingTransport', 'dryclean', 'DAGIP']

results = {method: {metric: [] for metric in METRICS} for method in METHODS}
binary_outcomes = {method: [] for method in METHODS}
continuous_outcomes = {method: [] for method in METHODS}
for dataset in DATASETS:

    data = np.load(os.path.join(RESULTS_FOLDER, f'{dataset}-baseline.json.npz'))
    Y_pred = data['ypred']
    Y_target = data['ytarget']
    pca = PCA()
    pca.fit(Y_target)
    xs = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = max(np.where(xs >= 0.95)[0][0] + 1, 5)
    #n_pcs = 45
    pca = PCA(n_components=n_pcs)
    pca.fit(Y_target)


    for method in METHODS:
        data = np.load(os.path.join(RESULTS_FOLDER, f'{dataset}-{method}.json.npz'))
        Y_pred = data['ypred']
        Y_target = data['ytarget']

        Y_target = pca.transform(Y_target)
        Y_pred = pca.transform(Y_pred)

        D = cdist(Y_pred, Y_target, metric='braycurtis')
        correct1 = (np.arange(D.shape[0]) == np.argmin(D, axis=0))
        correct2 = (np.arange(D.shape[0]) == np.argmin(D, axis=1))
        correct = np.logical_and(correct1, correct2)
        binary_outcomes[method].append(correct)
        results[method]['accuracy'].append(float(np.mean(correct)))

        ss_tot = np.mean(np.square(Y_target - np.mean(Y_target, axis=0)[np.newaxis, :]))
        ss_res = np.mean(np.square(Y_pred - Y_target))
        r2 = 1. - ss_res / ss_tot
        continuous_outcomes[method].append((ss_tot, ss_res, Y_target.size))
        results[method]['r2'].append(float(r2))


def one_sided_mcnemar_test(correct: np.ndarray, correct_dagip: np.ndarray) -> float:
    dagip_better = np.sum(np.logical_and(~correct, correct_dagip))
    dagip_worse = np.sum(np.logical_and(correct, ~correct_dagip))
    cont = np.asarray([[0, dagip_worse], [dagip_better, 0]])
    p_value = mcnemar(cont).pvalue
    if dagip_better > dagip_worse:
        p_value = p_value / 2.
    else:
        p_value = 1 - p_value / 2.
    return p_value


def z_test(r2_1: float, n_1: int, r2_2: float, n_2: int) -> float:
    r_1 = np.sqrt(np.clip(r2_1, 0, 1))
    r_2 = np.sqrt(np.clip(r2_2, 0, 1))
    num = 0.5 * np.log((1 + r_1) / (1 - r_1)) - 0.5 * np.log((1 + r_2) / (1 - r_2))
    den = np.sqrt(1. / (n_1 - 3) + 1. / (n_2 - 3))
    z = num / den
    return 1 - scipy.stats.norm.cdf(z)


ss_tot_dagip_total, ss_res_dagip_total, n_dagip_total = 0, 0, 0
for k in range(len(DATASETS)):
    ss_tot_dagip_total += continuous_outcomes['da'][k][0]
    ss_res_dagip_total += continuous_outcomes['da'][k][1]
    n_dagip_total += continuous_outcomes['da'][k][2]
for i, method in enumerate(METHODS):
    for k in range(len(DATASETS)):
        p_value = one_sided_mcnemar_test(binary_outcomes[method][k], binary_outcomes['da'][k])
        results[method]['mcnemar-pvalue'].append(p_value)
        ss_tot, ss_res, n = continuous_outcomes[method][k]
        ss_tot_dagip, ss_res_dagip, n_dagip = continuous_outcomes['da'][k]
        p_value = z_test(1 - ss_res_dagip / ss_tot_dagip, n_dagip, 1 - ss_res / ss_tot, n)
        results[method]['ztest-pvalue'].append(float(p_value))
    p_value = one_sided_mcnemar_test(
        np.concatenate([binary_outcomes[method][k] for k in range(len(DATASETS))], axis=0),
        np.concatenate([binary_outcomes['da'][k] for k in range(len(DATASETS))], axis=0),
    )
    results[method]['mcnemar-pvalue'].append(float(p_value))
    ss_tot_total, ss_res_total, n_total = 0, 0, 0
    for k in range(len(DATASETS)):
        ss_tot_total += continuous_outcomes[method][k][0]
        ss_res_total += continuous_outcomes[method][k][1]
        n_total += continuous_outcomes[method][k][2]
    p_value = z_test(1 - ss_res_dagip_total / ss_tot_dagip_total, n_dagip_total, 1 - ss_res / ss_tot, n)
    results[method]['ztest-pvalue'].append(float(p_value))


plt.figure(figsize=(12, 5))
for k in range(2):

    metric = METRICS[k + 2] if (SIGNIFICANCE) else METRICS[k]

    ax = plt.subplot(1, 2, k + 1)

    data = []
    for method in METHODS:
        row = results[method][metric]
        if (not SIGNIFICANCE):
            row.append(np.mean(row))
        data.append(row)
    data = np.asarray(data)

    index = [
        r'$\mathcal{D}_{9}$ protocol / $\mathcal{D}_{10}$ protocol',
        r'NovaSeq V1 chemistry ($\mathcal{D}_{6,a}$) / NovaSeq V1.5 chemistry ($\mathcal{D}_{6,b}$)',
        r'TruSeq Nano kit ($\mathcal{D}_{1,a}$) / Kapa HyperPrep kit ($\mathcal{D}_{1,b}$)',
        r'IDT indexes ($\mathcal{D}_{2,a}$) / Kapa dual indexes ($\mathcal{D}_{2,b}$)',
        r'HiSeq 2000 ($\mathcal{D}_{3,a}$) / NovaSeq ($\mathcal{D}_{3,b}$)',
        r'HiSeq 2500 ($\mathcal{D}_{4,a}$) / NovaSeq ($\mathcal{D}_{4,b}$)',
        r'HiSeq 4000 ($\mathcal{D}_{5,a}$) / NovaSeq ($\mathcal{D}_{5,b}$)',
        'Average'
    ]

    df = pd.DataFrame(data.T, index=index, columns=METHOD_NAMES)
    seaborn.heatmap(
        df, annot=True, fmt='.3f',
        cbar=False, linewidths=2, linecolor='white',
        yticklabels=(k == 0),
        ax=ax
    )
    ax.set_title(METRIC_NAMES[k])
plt.tight_layout()
if SIGNIFICANCE:
    plt.savefig(os.path.join(FIGURES_FOLDER, 'pairs-heatmaps-significance.png'), dpi=400)
else:
    plt.savefig(os.path.join(FIGURES_FOLDER, 'pairs-heatmaps.png'), dpi=400)
plt.show()
