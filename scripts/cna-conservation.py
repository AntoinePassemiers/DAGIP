import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import tqdm
import numpy as np
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import seaborn
from pycirclize import Circos
from pycirclize.utils import load_eukaryote_example_dataset

import ot.da
from dagip.segmentation import SovRefine
from dagip.core import ot_da, DomainAdapter
from dagip.correction.gc import gc_correction
from dagip.ichorcna.metrics import ploidy_accuracy, cna_accuracy, sign_accuracy, sov_refine, absolute_error
from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction import Positive
from dagip.tools.dryclean import run_dryclean
from dagip.tools.ichor_cna import ichor_cna, create_ichor_cna_normal_panel, load_ichor_cna_results


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
ICHORCNA_LOCATION = os.path.join(ROOT, '..', 'ichorCNA-master')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'corrected')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')

REFERENCE_FREE = False
METHOD = 'dagip'


PRETTY_NAMES = {
    'baseline': 'Baseline',
    'centering-scaling': 'Center-and-scale',
    'mapping-transport': 'MappingTransport',
    'dryclean': 'dryclean',
    'dagip': 'DAGIP'
}


# Load reference GC content and mappability
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc-content-1000kb.csv'))
gc_content = df['MEAN'].to_numpy()
bin_chr_names = df['CHR'].to_numpy()
bin_starts = df['START'].to_numpy()
bin_ends = df['END'].to_numpy()
df = pd.read_csv(os.path.join(DATA_FOLDER, 'mappability-1000kb.csv'))
mappability = df['MEAN'].to_numpy()

# Load data
data = np.load(os.path.join(DATA_FOLDER, 'numpy', 'OV.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
X = data['X']
y = (data['y'] == 'OV').astype(int)
d = (data['d'] == 'D9').astype(int)
paired_with = data['paired_with']
assert not np.any(np.isnan(X))

# Load sample pairs
gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
idx1_pairs, idx2_pairs = [], []
for i in range(len(X)):
    if d[i] == 0:
        continue
    if paired_with[i]:
        j = gc_code_dict[paired_with[i]]
        if y[i] == 1:
            idx1_pairs.append(i)
            idx2_pairs.append(j)
idx1_pairs = np.asarray(idx1_pairs, dtype=int)
idx2_pairs = np.asarray(idx2_pairs, dtype=int)
idx_pairs = set(list(np.concatenate((idx1_pairs, idx2_pairs), axis=0)))
print(f'Number of pairs: {len(idx1_pairs)}')

idx_y0d0 = np.asarray(list(set(np.where(np.logical_and(y == 0, d == 0))[0]).difference(idx_pairs)), dtype=int)
idx_y0d1 = np.asarray(list(set(np.where(np.logical_and(y == 0, d == 1))[0]).difference(idx_pairs)), dtype=int)
idx_y1d0 = np.asarray(list(set(np.where(np.logical_and(y == 1, d == 0))[0]).difference(idx_pairs)), dtype=int)
idx_y1d1 = np.asarray(list(set(np.where(np.logical_and(y == 1, d == 1))[0]).difference(idx_pairs)), dtype=int)
idx_d0 = np.asarray(list(set(np.where(d == 0)[0]).difference(idx_pairs)), dtype=int)
idx_d1 = np.asarray(list(set(np.where(d == 1)[0]).difference(idx_pairs)), dtype=int)

print(len(idx_y0d0), len(idx_y0d1), len(idx_y1d0), len(idx_y1d1))

# GC-correction
X = np.clip(X, 0, None)
X = gc_correction(X, gc_content)

# Domain adaptation
X_adapted = np.copy(X)
if METHOD == 'dagip':
    if not os.path.exists(os.path.join(RESULTS_FOLDER, 'dagip', f'OVi-corrected.npy')):
        folder = os.path.join(ROOT, 'tmp', 'ot-da-tmp')
        adapter = DomainAdapter(folder=folder, manifold=Positive())
        adapter.fit(
            [X_adapted[idx_y0d1, :], X_adapted[idx_y1d1, :]],
            [X_adapted[idx_y0d0, :], X_adapted[idx_y1d0, :]]
        )
        X_adapted[d == 1, :] = adapter.transform(X_adapted[d == 1, :])
        np.save(os.path.join(RESULTS_FOLDER, 'dagip', f'OVi-corrected.npy'), X_adapted)
    else:
        X_adapted = np.load(os.path.join(RESULTS_FOLDER, 'dagip', f'OVi-corrected.npy'))
elif METHOD == 'centering-scaling':

    # Correct controls
    target_scaler = RobustScaler()
    target_scaler.fit(X_adapted[idx_y0d0, :])
    source_scaler = RobustScaler()
    source_scaler.fit(X_adapted[idx_y0d1, :])
    X_adapted[np.logical_and(y == 0, d == 1), :] = target_scaler.inverse_transform(source_scaler.transform(X_adapted[np.logical_and(y == 0, d == 1), :]))

    # Correct casses
    target_scaler = RobustScaler()
    target_scaler.fit(X_adapted[idx_y1d0, :])
    source_scaler = RobustScaler()
    source_scaler.fit(X_adapted[idx_y1d1, :])
    X_adapted[np.logical_and(y == 1, d == 1), :] = target_scaler.inverse_transform(source_scaler.transform(X_adapted[np.logical_and(y == 1, d == 1), :]))

elif METHOD == 'baseline':
    pass  # No correction
elif METHOD == 'dryclean':
    if not os.path.exists(os.path.join(RESULTS_FOLDER, 'dryclean', f'OVi-corrected.npy')):
        X_adapted[d == 0, :] = run_dryclean(bin_chr_names, bin_starts, bin_ends,
            X_adapted[np.logical_and(d == 0, y == 0), :], X_adapted[d == 0, :], 'tmp-dryclean')
        X_adapted[d == 1, :] = run_dryclean(bin_chr_names, bin_starts, bin_ends,
            X_adapted[np.logical_and(d == 1, y == 0), :], X[d == 1, :], 'tmp-dryclean')
        np.save(os.path.join(RESULTS_FOLDER, 'dryclean', f'OVi-corrected.npy'), X_adapted)
    else:
        X_adapted = np.load(os.path.join(RESULTS_FOLDER, 'dryclean', f'OVi-corrected.npy'))
    X = X_adapted
elif METHOD == 'mapping-transport':
    model = ot.da.MappingTransport()
    model.fit(
        Xs=X_adapted[idx_d1, :],
        ys=y[idx_d1],
        Xt=X_adapted[idx_d0, :]
    )
    X_adapted[d == 1, :] = model.transform(Xs=X_adapted[d == 1, :])
else:
    raise NotImplementedError(f'Unknown correction method "{METHOD}"')

# Define how to create panel of normals
def create_pon(pon_folder: str, X_sub: np.ndarray) -> None:
    if not os.path.exists(os.path.join(ROOT, 'ichor-cna-results', 'normal-panels', pon_folder, 'normal-panel_median.rds')):
        create_ichor_cna_normal_panel(
            ICHORCNA_LOCATION,
            os.path.join(ROOT, 'ichor-cna-results', 'normal-panels', pon_folder),
            X_sub, gc_content, mappability
        )

# Create panels of normals
if not REFERENCE_FREE:
    create_pon('d1', X[np.logical_and(d == 1, y == 0), :])
    create_pon('d0', X[np.logical_and(d == 0, y == 0), :])
    create_pon(os.path.join('d1-adapted', METHOD), X_adapted[np.logical_and(d == 1, y == 0), :])


# Define how to call CNAs in cancer samples
def cna_calling(pon_folder: str, res_folder: str, X_sub: np.ndarray, gc_codes_: np.ndarray, reference_free: bool = False) -> None:
    all_results = []
    for i in range(len(gc_codes_)):
        if reference_free:
            folder = os.path.join('ichor-cna-results', 'noref', res_folder, gc_codes_[i])
        else:
            folder = os.path.join('ichor-cna-results', res_folder, gc_codes_[i])
        os.makedirs(folder, exist_ok=True)
        results = load_ichor_cna_results(folder)
        normal_panel_filepath = os.path.join(ROOT, 'ichor-cna-results', 'normal-panels', pon_folder, 'normal-panel_median.rds')
        if not results['success']:
            ichor_cna(
                ICHORCNA_LOCATION,
                None if REFERENCE_FREE else normal_panel_filepath,
                folder,
                X_sub[i, :],
                gc_content,
                mappability
            )
            results = load_ichor_cna_results(folder)
        all_results.append(results)
    return all_results


results_1_1 = cna_calling('d1', 'd1', X[idx1_pairs, :], gc_codes[idx1_pairs], reference_free=REFERENCE_FREE)
results_1a_1a = cna_calling(os.path.join('d1-adapted', METHOD), os.path.join('d1-adapted', 'd1-controls', METHOD), X_adapted[idx1_pairs, :], gc_codes[idx1_pairs], reference_free=REFERENCE_FREE)
results_0_0 = cna_calling('d0', 'd0', X[idx2_pairs, :], gc_codes[idx2_pairs], reference_free=REFERENCE_FREE)
results_0_1a = cna_calling('d0', os.path.join('d1-adapted', 'd0-controls', METHOD), X_adapted[idx1_pairs, :], gc_codes[idx1_pairs], reference_free=REFERENCE_FREE)


def compute_differences(res1, res2):
    sovs = np.asarray([SovRefine(x['copy-number'], y['copy-number']).sov_refine() for x, y in zip(res1, res2)])
    cna1 = np.asarray([x['copy-number'] for x in res1])
    cna2 = np.asarray([x['copy-number'] for x in res2])
    log_r1 = np.asarray([x['log-r'] for x in res1], dtype=float)
    log_r2 = np.asarray([x['log-r'] for x in res2], dtype=float)
    fractions1 = np.asarray([x['tumor-fraction'] for x in res1], dtype=float)
    fractions2 = np.asarray([x['tumor-fraction'] for x in res2], dtype=float)
    ploidy1 = np.asarray([x['tumor-ploidy'] for x in res1], dtype=float)
    ploidy2 = np.asarray([x['tumor-ploidy'] for x in res2], dtype=float)
    prevalence1 = np.asarray([x['tumor-cellular-prevalence'] for x in res1], dtype=float)
    prevalence2 = np.asarray([x['tumor-cellular-prevalence'] for x in res2], dtype=float)
    mask = ~np.logical_or(np.isnan(prevalence1), np.isnan(prevalence2))
    prevalence1, prevalence2 = prevalence1[mask], prevalence2[mask]
    subclonal1 = np.asarray([x['proportion-subclonal-cnas'] for x in res1], dtype=float)
    subclonal2 = np.asarray([x['proportion-subclonal-cnas'] for x in res2], dtype=float)
    mask = ~np.logical_or(np.isnan(subclonal1), np.isnan(subclonal2))
    subclonal1, subclonal2 = subclonal1[mask], subclonal2[mask]
    return {
        'cna': np.mean(cna1 == cna2, axis=1),
        'cna-sign': np.mean(np.sign(cna1) == np.sign(cna2), axis=1),
        'SOV_REFINE': sovs,
        'log-r': np.mean(np.abs(log_r1 - log_r2), axis=1),
        'tf': np.abs(fractions1 - fractions2),
        'ploidy': np.abs(ploidy1 - ploidy2),
        'prevalence': np.abs(prevalence1 - prevalence2),
        'subclonal': np.abs(subclonal1 - subclonal2)
    }


def compute_metrics_signifiance(res1, res2, res1_baseline, res2_baseline):
    res = compute_differences(res1, res2)
    res_baseline = compute_differences(res1_baseline, res2_baseline)

    return {
        'Accuracy (ploidy)': scipy.stats.ttest_rel(res['cna'], res_baseline['cna'], alternative='less').pvalue,
        'Accuracy (ploidy sign)': scipy.stats.ttest_rel(res['cna-sign'], res_baseline['cna-sign'], alternative='less').pvalue,
        'SOV_REFINE': scipy.stats.ttest_rel(res['SOV_REFINE'], res_baseline['SOV_REFINE'], alternative='less').pvalue,
        'Log-ratios': scipy.stats.ttest_rel(res['log-r'], res_baseline['log-r'], alternative='greater').pvalue,
        'Tumor fraction': scipy.stats.ttest_rel(res['tf'], res_baseline['tf'], alternative='greater').pvalue,
        'Tumor ploidy': scipy.stats.ttest_rel(res['ploidy'], res_baseline['ploidy'], alternative='greater').pvalue,
        'Cellular prevalence': scipy.stats.ttest_ind(res['prevalence'], res_baseline['prevalence'], alternative='greater').pvalue,
        'Proportion of subclonal CNAs': scipy.stats.ttest_ind(res['subclonal'], res_baseline['subclonal'], alternative='greater').pvalue,
    }


def compute_metrics(res1, res2):

    cna1 = [x['copy-number'] for x in res1]
    cna2 = [x['copy-number'] for x in res2]
    log_r1 = np.asarray([x['log-r'] for x in res1], dtype=float)
    log_r2 = np.asarray([x['log-r'] for x in res2], dtype=float)
    fractions1 = np.asarray([x['tumor-fraction'] for x in res1], dtype=float)
    fractions2 = np.asarray([x['tumor-fraction'] for x in res2], dtype=float)
    ploidy1 = np.asarray([x['tumor-ploidy'] for x in res1], dtype=float)
    ploidy2 = np.asarray([x['tumor-ploidy'] for x in res2], dtype=float)

    prevalence1 = np.asarray([x['tumor-cellular-prevalence'] for x in res1], dtype=float)
    prevalence2 = np.asarray([x['tumor-cellular-prevalence'] for x in res2], dtype=float)
    mask = ~np.logical_or(np.isnan(prevalence1), np.isnan(prevalence2))
    prevalence1, prevalence2 = prevalence1[mask], prevalence2[mask]

    subclonal1 = np.asarray([x['proportion-subclonal-cnas'] for x in res1], dtype=float)
    subclonal2 = np.asarray([x['proportion-subclonal-cnas'] for x in res2], dtype=float)
    mask = ~np.logical_or(np.isnan(subclonal1), np.isnan(subclonal2))
    subclonal1, subclonal2 = subclonal1[mask], subclonal2[mask]

    metrics = {
        'Accuracy (ploidy)': np.mean(ploidy_accuracy(cna1, cna2)),
        'Accuracy (ploidy sign)': np.mean(sign_accuracy(cna1, cna2)),
        'SOV_REFINE': np.mean(sov_refine(cna1, cna2)),
        'r2 (Log ratios)': r2_score(log_r1.flatten(), log_r2.flatten()),
        'r2 (tumor fractions)': r2_score(fractions1.flatten(), fractions2.flatten()),
        'r2 (tumor ploidy)': r2_score(ploidy1.flatten(), ploidy2.flatten()),
        'r2 (cellular prevalence)': r2_score(prevalence1.flatten(), prevalence2.flatten()),
        'r2 (proportion of subclonal CNAs)': r2_score(subclonal1.flatten(), subclonal2.flatten())
    }
    metrics['Average'] = np.mean(list(metrics.values()))

    print(metrics)
    print('')

    return metrics


metrics_1_1 = compute_metrics(results_1_1, results_1a_1a)
metrics_0_0 = compute_metrics(results_0_0, results_0_1a)
print(f'Score 1: {np.mean(list(metrics_1_1.values()))}')
print(f'Score 2: {np.mean(list(metrics_0_0.values()))}')
print(f'Overall score: {0.5 * np.mean(list(metrics_1_1.values())) + 0.5 * np.mean(list(metrics_0_0.values()))}')


plt.figure(figsize=(4, 4))
ax = plt.subplot(1, 1, 1)
for reference_free in [True, False]:
    results_1_1 = cna_calling('d1', 'd1', None, gc_codes[idx1_pairs], reference_free=reference_free)
    results_1a_1a = cna_calling(os.path.join('d1-adapted', METHOD), os.path.join('d1-adapted', 'd1-controls', METHOD), None, gc_codes[idx1_pairs], reference_free=reference_free)
    results_0_0 = cna_calling('d0', 'd0', None, gc_codes[idx2_pairs], reference_free=reference_free)
    results_0_1a = cna_calling('d0', os.path.join('d1-adapted', 'd0-controls', METHOD), None, gc_codes[idx1_pairs], reference_free=reference_free)

    kwargs = dict(marker='v', s=40, alpha=0.6)
    kwargs['color'] = ('goldenrod' if reference_free else 'teal')
    kwargs['label'] = ('Reference-free' if reference_free else 'Reference-based')
    xs = [res['tumor-fraction'] for res in results_0_0]
    ys = [np.nanmean(res1['copy-number'] == res2['copy-number']) for res1, res2 in zip(results_0_0, results_0_1a)]
    ax.scatter(xs, ys, **kwargs)

    print(reference_free, scipy.stats.pearsonr(xs, ys))

ax.legend()
ax.set_xlabel('Estimated TF before correction: Setting (3)')
ax.set_ylabel('Copy number consistency between \nsettings (3) and (4): Accuracy')
for side in ['right', 'top']:
    ax.spines[side].set_visible(False)
ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'tf-vs-acc.png'), dpi=400)
#plt.show()


plt.figure(figsize=(4, 4))
ax = plt.subplot(1, 1, 1)
for reference_free in [True, False]:
    results_1_1 = cna_calling('d1', 'd1', None, gc_codes[idx1_pairs], reference_free=reference_free)
    results_1a_1a = cna_calling(os.path.join('d1-adapted', METHOD), os.path.join('d1-adapted', 'd1-controls', METHOD), None, gc_codes[idx1_pairs], reference_free=reference_free)
    results_0_0 = cna_calling('d0', 'd0', None, gc_codes[idx2_pairs], reference_free=reference_free)
    results_0_1a = cna_calling('d0', os.path.join('d1-adapted', 'd0-controls', METHOD), None, gc_codes[idx1_pairs], reference_free=reference_free)

    print(results_1_1[0])
    kwargs = dict(marker='v', s=40, alpha=0.6)
    kwargs['color'] = ('goldenrod' if reference_free else 'teal')
    kwargs['label'] = ('Reference-free' if reference_free else 'Reference-based')
    xs = [res['tumor-fraction'] for res in results_0_0]
    ys = [res['tumor-fraction'] for res in results_0_1a]
    ax.scatter(xs, ys, **kwargs)

    print(reference_free, scipy.stats.pearsonr(xs, ys))

ax.legend()
ax.set_xlabel('Estimated TF before correction: Setting (3)')
ax.set_ylabel('Estimated TF after correction: Setting (4)')
ax.set_title('Tumor fractions')
for side in ['right', 'top']:
    ax.spines[side].set_visible(False)
ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
ax.plot([0.001, 0.15], [0.001, 0.15], linestyle='--', linewidth=0.5, color='black')
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'tf.png'), dpi=400)
#plt.show()


plt.figure(figsize=(12, 12))
for k in range(2):

    ax = plt.subplot(2, 2, k + 1)

    COLUMNS = ['Accuracy (copy number)', 'Accuracy (copy number sign)', 'SOV_REFINE', r'$R^2$ (log-ratios)', r'$R^2$ (tumor fractions)', r'$R^2$ (tumor ploidy)', r'$R^2$ (cellular prevalence)', r'$R^2$ (proportion of subclonal CNAs)', 'Average']
    METHODS = ['baseline', 'centering-scaling', 'mapping-transport', 'dryclean', 'dagip']
    #METHODS = ['dagip']
    data = []
    for method in METHODS:
        reference_free = False
        results_1_1 = cna_calling('d1', 'd1', None, gc_codes[idx1_pairs], reference_free=reference_free)
        results_1a_1a = cna_calling(os.path.join('d1-adapted', method), os.path.join('d1-adapted', 'd1-controls', method), None, gc_codes[idx1_pairs], reference_free=reference_free)
        results_0_0 = cna_calling('d0', 'd0', None, gc_codes[idx2_pairs], reference_free=reference_free)
        results_0_1a = cna_calling('d0', os.path.join('d1-adapted', 'd0-controls', method), None, gc_codes[idx1_pairs], reference_free=reference_free)
        metrics_1_1 = compute_metrics(results_1_1, results_1a_1a)
        metrics_0_0 = compute_metrics(results_0_0, results_0_1a)
        if k == 0:
            data.append(np.asarray(list(metrics_1_1.values()), dtype=float))
            ax.set_title('Consistency between ichorCNA runs (1) and (2):\nIntra-domain consistency')
        else:
            data.append(np.asarray(list(metrics_0_0.values()), dtype=float))
            ax.set_title('Consistency between ichorCNA runs (3) and (4):\nCross-domain consistency')
    data = np.asarray(data)

    df = pd.DataFrame(
        np.clip(data.T, 0, None),
        index=COLUMNS,
        columns=[PRETTY_NAMES[method] for method in METHODS]
    )
    annot = pd.DataFrame(
        data.T,
        index=COLUMNS,
        columns=[PRETTY_NAMES[method] for method in METHODS]
    )
    seaborn.heatmap(
        df, annot=annot, fmt='.3f', ax=ax, yticklabels=(k == 0),
        cbar=False, linewidths=2, linecolor='white'
    )


    COLUMNS = ['Accuracy (copy number)', 'Accuracy (copy number sign)', 'SOV_REFINE', r'$R^2$ (log-ratios)', r'$R^2$ (tumor fractions)', r'$R^2$ (tumor ploidy)', r'$R^2$ (cellular prevalence)', r'$R^2$ (proportion of subclonal CNAs)']
    results_1a_1a_baseline = cna_calling(os.path.join('d1-adapted', 'dagip'), os.path.join('d1-adapted', 'd1-controls', 'dagip'), None, gc_codes[idx1_pairs], reference_free=reference_free)
    results_0_1a_baseline = cna_calling('d0', os.path.join('d1-adapted', 'd0-controls', 'dagip'), None, gc_codes[idx1_pairs], reference_free=reference_free)

    ax = plt.subplot(2, 2, k + 3)

    data = []
    for method in METHODS:
        reference_free = False
        results_1_1 = cna_calling('d1', 'd1', None, gc_codes[idx1_pairs], reference_free=reference_free)
        results_1a_1a = cna_calling(os.path.join('d1-adapted', method), os.path.join('d1-adapted', 'd1-controls', method), None, gc_codes[idx1_pairs], reference_free=reference_free)
        results_0_0 = cna_calling('d0', 'd0', None, gc_codes[idx2_pairs], reference_free=reference_free)
        results_0_1a = cna_calling('d0', os.path.join('d1-adapted', 'd0-controls', method), None, gc_codes[idx1_pairs], reference_free=reference_free)
        p_values_1_1 = np.asarray(list(compute_metrics_signifiance(results_1_1, results_1a_1a, results_1_1, results_1a_1a_baseline).values()))
        p_values_0_0 = np.asarray(list(compute_metrics_signifiance(results_0_0, results_0_1a, results_0_0, results_0_1a_baseline).values()))
        if method == 'dagip':
            p_values_1_1[:] = np.nan
            p_values_0_0[:] = np.nan


        if k == 0:
            data.append(p_values_1_1)
            ax.set_title('Intra-domain consistency\nSignificance of improvement of DAGIP\nover each method (t-test p-value)')
        else:
            data.append(p_values_0_0)
            ax.set_title('Cross-domain consistency\nSignificance of improvement of DAGIP\nover each method (t-test p-value)')
    data = np.asarray(data)

    df = pd.DataFrame(
        np.clip(data.T, 0, None),
        index=COLUMNS,
        columns=[PRETTY_NAMES[method] for method in METHODS]
    )
    seaborn.heatmap(
        df, annot=True, fmt='.3f', ax=ax, yticklabels=(k == 0),
        cbar=False, linewidths=2, linecolor='white', cmap='viridis_r'
    )


plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'ichorcna-heatmaps.png'), dpi=400)
plt.show()


import sys; sys.exit(0)

chr_bed_file, cytoband_file, _ = load_eukaryote_example_dataset('hg38')

results_1_1 = cna_calling('d1', 'd1', None, gc_codes[idx1_pairs], reference_free=False)
results_1a_1a = cna_calling(os.path.join('d1-adapted', METHOD), os.path.join('d1-adapted', 'd1-controls', METHOD), None, gc_codes[idx1_pairs], reference_free=False)
results_0_0 = cna_calling('d0', 'd0', None, gc_codes[idx2_pairs], reference_free=False)
results_0_1a = cna_calling('d0', os.path.join('d1-adapted', 'd0-controls', METHOD), None, gc_codes[idx1_pairs], reference_free=False)

tumour_fractions = np.asarray([x['tumor-fraction'] for x in results_1_1])
idx = np.argsort(tumour_fractions)

cna3 = np.asarray([x['copy-number'] for x in results_0_0])[idx, :]
cna4 = np.asarray([x['copy-number'] for x in results_0_1a])[idx, :]
cna1 = np.asarray([x['copy-number'] for x in results_1_1])[idx, :]
cna2 = np.asarray([x['copy-number'] for x in results_1a_1a])[idx, :]

CHR_SIZES = [250, 244, 200, 192, 183, 172, 161, 147, 140, 135, 137, 135, 116, 109, 103, 92, 85, 82, 60, 66, 48, 30]
STARTS = np.asarray([0] + list(np.cumsum(CHR_SIZES)))
STARTS, ENDS = STARTS[:-1], STARTS[1:]
print(X.shape[1], sum(CHR_SIZES))
assert X.shape[1] == sum(CHR_SIZES)

# Plot circos plot
chr_bed_file = os.path.join(DATA_FOLDER, 'hg38_chr.bed')
cytoband_file = os.path.join(DATA_FOLDER, 'hg38_cytoband.tsv')
circos = Circos.initialize_from_bed(chr_bed_file, space=3)
circos.add_cytoband_tracks((95, 100), cytoband_file)
for start, end, sector in tqdm.tqdm(list(zip(STARTS, ENDS, circos.sectors)), desc='Plotting CNAs'):
    sector.text(sector.name, size=10)

    track1 = sector.add_track((60, 90), r_pad_ratio=0.1)
    track1.axis()
    track1.heatmap(
        np.sign(cna4 - cna3)[:, start:end],
        cmap='bwr', # RdBu
        vmin=-1,
        vmax=1
    )

    track2 = sector.add_track((25, 55), r_pad_ratio=0.1)
    track2.axis()
    track2.heatmap(
        np.sign(cna2 - cna1)[:, start:end],
        cmap='PiYG',
        vmin=-1,
        vmax=1
    )

fig = circos.plotfig()

plt.title(PRETTY_NAMES[METHOD])

# Save figure
os.makedirs(FIGURES_FOLDER, exist_ok=True)
plt.savefig(os.path.join(FIGURES_FOLDER, f'ichorcna-{METHOD}.png'), dpi=400)
plt.show()

print('Finished.')