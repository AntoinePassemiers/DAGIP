from typing import List

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.svm import SVC

from dagip.nipt.binning import ChromosomeBounds


MARKERS = ['x', 'o', 's', 'P', 'D', 'h', '*', 'p', '8']



def get_parameters(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = SVC()
    model.fit(X, y)
    alpha = np.zeros(len(X))
    alpha[model.support_] = model.dual_coef_
    return alpha


def loo_influence_analysis(X: np.ndarray, y: np.ndarray) -> np.ndarray:

    # Train model on all samples
    #X = RobustScaler().fit_transform(X)
    base_params = get_parameters(X, y)

    # Leave-one-out
    scores = np.zeros(len(X))
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        params = np.zeros(len(X))
        params[mask] = get_parameters(X[mask, :], y[mask])
        scores[i] = np.mean(np.abs(base_params - params))
    return scores


def scatterplot_with_sample_importances(ax, axl, X, y, d, labels, cancer_stages, style_dict, stage0_label='Stage 0', legend: bool = False):

    labels = np.copy(labels)

    palette = {key: value[0] for key, value in style_dict.items()}
    markers = {key: value[1] for key, value in style_dict.items()}

    for i in range(len(labels)):
        key = labels[i]
        if cancer_stages[i] == '0':
            labels[i] += '-stage0'
            palette[key + '-stage0'] = palette[key]
            markers[key + '-stage0'] = 'X'

    coords = TSNE().fit_transform(X)

    sizes = [39, 40, 90, 140, 200]
    size_labels = [stage0_label, 'Stage I', 'Stage II', 'Stage III', 'Stage IV']

    cancer_stage_info_available = False
    importances = np.full(len(X), sizes[0])
    for k, stage in enumerate(['0', 'I', 'II', 'III', 'IV']):
        importances[cancer_stages == stage] = sizes[k]
        if np.any(cancer_stages == stage):
            cancer_stage_info_available = True

    idx = np.arange(len(coords))
    np.random.shuffle(idx)  # Shuffle display order to ensure no class is dominating the visual space
    g = seaborn.scatterplot(
        ax=ax,
        x=coords[idx, 0],
        y=coords[idx, 1],
        hue=labels[idx],
        hue_order=list(palette.keys()),
        style=labels[idx],
        size=importances[idx],
        sizes=(min(sizes), max(sizes)),
        palette=palette,
        markers=markers,
        legend=False
    )
    ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
    ax.set_xlabel('First t-SNE component', fontsize=15)
    ax.set_ylabel('Second t-SNE component', fontsize=15)

    # Custom legend
    if legend:
        handles, labels = [], []

        # Marker color labels
        for label, (color, marker) in style_dict.items():
            handles.append(plt.scatter([], [], s=sizes[0], color=color, marker=marker))
            labels.append(label)

        if cancer_stage_info_available:

            # Add white space
            handles.append(plt.scatter([], [], s=0))
            labels.append(' ')

            # Marker size labels
            if np.any(cancer_stages == '0'):
                handles.append(plt.scatter([], [], s=sizes[0], color='black', marker='x'))
                labels.append(stage0_label)
            for size, label in zip(sizes[1:], size_labels[1:]):
                handles.append(plt.scatter([], [], s=size, color='black'))
                labels.append(label)
        
        axl.legend(
            handles=handles,
            labels=labels,
            prop={'size': 10},
            #bbox_to_anchor=(1.1, 1.05)
            loc='center left'
        )


def plot_end_motif_freqs(ax, x: np.ndarray):
    assert x.shape == (256,)
    x = x / np.sum(x)
    xs, ys = np.arange(len(x)) + 0.5, x
    plt.bar(xs[:64], ys[:64], color='royalblue')
    plt.bar(xs[64:128], ys[64:128], color='gold')
    plt.bar(xs[128:192], ys[128:192], color='coral')
    plt.bar(xs[192:], ys[192:], color='mediumseagreen')
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
    ax.set_xticks(range(0, 257, 16), minor=True)
    ax.set_xticklabels(['' for _ in range(17)], minor=True)
    ax.set_xticks(np.arange(0, 256, 16) + 8, minor=False)
    labels = ['AANN', 'ATNN', 'ACNN', 'AGNN', 'TANN', 'TTNN', 'TCNN', 'TGNN', 'CANN', 'CTNN', 'CCNN', 'CGNN', 'GANN', 'GTNN', 'GCNN', 'GGNN']
    ax.set_xticklabels(labels, minor=False)
    ax.grid(which='minor', alpha=0.8, linestyle='--', linewidth=0.8, color='black')
    plt.axhline(y=0, linestyle='--', color='black', linewidth=0.5)
    ax.set_ylabel('5\' end-motif frequency', fontsize=15)


def plot_ot_plan_degrees(ax, gamma: np.ndarray, eps: float = 1e-7):
    unique, counts = np.unique(np.sum(gamma > eps, axis=0), return_counts=True)
    ax.bar(
        unique - 0.15, counts, color='darkgoldenrod', width=0.3,
        label='Source domain'
    )
    unique2, counts = np.unique(np.sum(gamma > eps, axis=1), return_counts=True)
    ax.bar(
        unique2 + 0.15, counts, color='darkcyan', width=0.3,
        label='Target domain'
    )
    ax.legend()
    ax.set_xticks(range(1, max(max(unique), max(unique2)) + 1), range(1, max(max(unique), max(unique2)) + 1))
    ax.set_xlabel(r'#Similar points in source domain')
    ax.set_ylabel(r'#Points in target domain')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def scatter_plot(X: np.ndarray, y: np.ndarray, d: np.ndarray):
    y = y.astype(float)
    coords = KernelPCA().fit_transform(X)
    for domain in np.unique(d):
        mask = (d == domain)
        marker = MARKERS[domain % len(MARKERS)]
        plt.scatter(coords[mask, 0], coords[mask, 1], alpha=0.5, c=y[mask], marker=marker)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')


def plot_gc_bias(X1_adapted: np.ndarray, X1: np.ndarray, gc_content: np.ndarray):
    x1_adapted = np.mean(X1_adapted, axis=0)
    x1 = np.mean(X1, axis=0)
    chr_bounds = ChromosomeBounds.get_10kb()
    ys = [np.mean(X1_adapted[:, i:i+1]) - np.mean(X1[:, i:i+1]) for i in range(22)]
    cmap = plt.get_cmap('bwr')
    mask = (np.random.rand(len(gc_content)) < 0.1)
    mask = np.logical_and(mask, np.logical_and(gc_content > 0.3, gc_content < 0.7))
    fig, ax = plt.subplots(figsize=(6.05, 5.5))
    mappable = ax.scatter(x1[mask], x1_adapted[mask], c=gc_content[mask], s=2, cmap=cmap)
    ax.plot([0, 1.2], [0, 1.2], color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Average normalised read counts')
    ax.set_ylabel('Average corrected read counts')
    fig.colorbar(mappable)
    plt.tight_layout()


"""
fig, ax = plt.subplots(2, 1, figsize=(4, 8))
coords = TSNE().fit_transform(ChromosomeBounds.bin_from_10kb_to_1mb(X))
ax[0].scatter(coords[idx1, 0], coords[idx1, 1], edgecolors='mediumseagreen', marker='s', c='None')
ax[0].scatter(coords[idx2, 0], coords[idx2, 1], edgecolors='royalblue', marker='o', c='None')
idx3 = np.logical_and(d == 0, y == 1)
ax[0].scatter(coords[idx3, 0], coords[idx3, 1], color='orchid', marker='x')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].set_xlabel('First component')
ax[0].set_ylabel('Second component')
coords = TSNE().fit_transform(ChromosomeBounds.bin_from_10kb_to_1mb(X_adapted))
ax[1].scatter(coords[idx1, 0], coords[idx1, 1], label='Controls (TruSeq ChIP)', edgecolors='mediumseagreen', marker='s', c='None')
ax[1].scatter(coords[idx2, 0], coords[idx2, 1], label='Controls (KAPA HyperPrep)', edgecolors='royalblue', marker='o', c='None')
idx3 = np.logical_and(d == 0, y == 1)
ax[1].scatter(coords[idx3, 0], coords[idx3, 1], label='Haematological cancer', color='orchid', marker='x')
ax[1].legend(prop={'size': 7})
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].set_xlabel('First component')
ax[1].set_ylabel('Second component')
plt.tight_layout()
plt.savefig('tsne.png', dpi=200)
"""
