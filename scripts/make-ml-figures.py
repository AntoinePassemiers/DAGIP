import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))

import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd
import statsmodels.stats.api as sms
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn

from dagip.utils import LaTeXTable


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'supervised-learning')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')


METHODS = ['baseline', 'dryclean', 'center-and-scale', 'kmm', 'mapping-transport', 'da']
METHOD_NAMES = ['Baseline', 'dryclean', 'Center-and-scale', 'KMM', 'MappingTransport', 'DAGIP']
#METHODS = ['baseline', 'center-and-scale', 'kmm', 'mapping-transport', 'da']
COLORS = ['#a883ef', '#f996ab', '#f35ee2', 'teal', '#f4c45a', '#b1e468']
palette = {method_name: color for method_name, color in zip(METHOD_NAMES, COLORS)}
TITLES = [
    'HL (coverage profiles)',
    'DLBCL (coverage profiles)',
    'MM (coverage profiles)',
    'OV (coverage profiles)',
    'BRCA (Multimodal)'
]
DATASETS = ['HL', 'DLBCL', 'MM', 'OV-forward', 'BRCA']


print('Computation time:')
for method_name in METHODS:
    row = []
    for dataset in DATASETS:
        if not os.path.exists(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json')):
            continue
        with open(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json'), 'r') as f:
            data = json.load(f)
            data = data[method_name]
            row.append(data['extra']['computation-time'])

    metric = np.mean(row)
    print(method_name, metric)

print('')

for dataset in DATASETS:
    table = LaTeXTable()
    for method_name, pretty_method_name in zip(METHODS, METHOD_NAMES):
        if not os.path.exists(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json')):
            continue
        with open(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json'), 'r') as f:
            data = json.load(f)
            data = data[method_name]
            table.add(method_name, data['supervised-learning'])

    print('')
    print(dataset)
    print(table)
    print('')



for metric in ['mcc', 'auroc']:

    all_results = {method_name: [] for method_name in METHOD_NAMES}

    df_data = {metric: [], 'Method': [], 'Pathology': []}
    error_bars = []
    for dataset in DATASETS:
        for method_name, pretty_method_name in zip(METHODS, METHOD_NAMES):
            if not os.path.exists(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json')):
                continue
            with open(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json'), 'r') as f:
                data = json.load(f)
                data = data[method_name]
            
            res = []
            for ml_model in ['rf', 'svm', 'reglog']:
                res += [x[metric] for x in data['supervised-learning'][ml_model]]
            all_results[pretty_method_name] += res
            confint = sms.DescrStatsW(res).tconfint_mean()
            df_data[metric].append(np.mean(res))
            df_data['Method'].append(pretty_method_name)
            df_data['Pathology'].append(dataset if (dataset != 'OV-forward') else 'OV')
            error_bars.append(0.5 * (confint[1] - confint[0]))

    for pretty_method_name in METHOD_NAMES:
        print(f'Average {metric} for {pretty_method_name}: {np.mean(all_results[pretty_method_name])}')

    df = pd.DataFrame(df_data)

    plt.figure(figsize=(4, 8))

    ax = plt.subplot(1, 1, 1)
    sns.barplot(ax=ax, x=metric, y='Pathology', hue='Method', data=df, palette=palette, orient='h', legend=False)
    ax.set_xlabel(metric.upper(), fontsize=20)
    ax.set_ylabel('Cancer type', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=17)
    sns.despine()

    for i, bar in enumerate(ax.patches):
        bar_height = bar.get_height()
        bar_y = bar.get_y()
        bar_width = bar.get_width()
        ax.errorbar(
            bar_width, bar_y + bar_height / 2, xerr=error_bars[i], fmt='none', 
            capsize=5, color='black'
        )

        fontsize = 9
        if bar_width - error_bars[i] < 0.2:
            ax.annotate(
                f'{bar_width:.3f} ± {error_bars[i]:.3f}', 
                (bar_width + error_bars[i], bar_y + bar_height / 2.), 
                ha='center',
                va='center',
                xytext=(40, -1),
                textcoords='offset points',
                color='black',
                weight='bold',
                fontsize=fontsize,
            )
        else:
            ax.annotate(
                f'{bar_width:.3f}', 
                (bar_width - error_bars[i], bar_y + bar_height / 2.), 
                ha='center',
                va='center',
                xytext=(-20, -1),
                textcoords='offset points',
                color='white',
                weight='bold',
                fontsize=fontsize,
            )

            ax.annotate(
                f'± {error_bars[i]:.3f}', 
                (bar_width + error_bars[i], bar_y + bar_height / 2.), 
                ha='center',
                va='center',
                xytext=(25, -1),
                textcoords='offset points',
                color='black',
                weight='bold',
                fontsize=fontsize,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f'{metric}.png'), dpi=400)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    mat = []
    for method1 in METHOD_NAMES:
        row = []
        for method2 in METHOD_NAMES:
            xs = all_results[method1]
            ys = all_results[method2]
            if len(xs) == len(ys):  # Methods were run on the same folds
                pvalue = scipy.stats.ttest_rel(xs, ys, alternative='less').pvalue
            else:
                pvalue = scipy.stats.ttest_ind(xs, ys, alternative='less').pvalue
            row.append(pvalue)
        mat.append(row)
    mat = np.asarray(mat)

    df = pd.DataFrame(
        mat,
        index=METHOD_NAMES,
        columns=METHOD_NAMES
    )
    seaborn.heatmap(
        df, annot=True, fmt='.1e', ax=ax,
        cbar=False, linewidths=2, linecolor='white', cmap='viridis_r'
    )
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_title(f'Significance of {metric.upper()} differences (t-test p-values)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f'{metric}-significance.png'), dpi=400)


plt.figure(figsize=(12, 8))
roc = True

for k, (title, dataset) in enumerate(zip(TITLES, DATASETS)):
    results = []

    ax = plt.subplot(2, 3, k + 1)

    for i, method_name in enumerate(METHODS):

        if (method_name == 'dryclean') and (dataset not in {'HL', 'DLBCL', 'MM', 'OV-forward'}):
            continue

        with open(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json'), 'r') as f:
            data = json.load(f)
            data = data[method_name]

        y, y_hat = [], []
        for ml_model in ['svm']:
            for res in data['supervised-learning'][ml_model]:
                y.append(res['y'])
                y_hat.append(res['y-pred'])
        y, y_hat = np.concatenate(y, axis=0), np.concatenate(y_hat, axis=0)

        fontsize = 15
        if roc:
            fpr, tpr, _ = roc_curve(y, y_hat)
            ax.plot(fpr, tpr, label=method_name, color=COLORS[i], alpha=0.8)
            ax.set_xlabel('False positive rate', fontsize=fontsize)
            ax.set_ylabel('True positive rate', fontsize=fontsize)
        else:
            precision, recall, _ = precision_recall_curve(y, y_hat)
            ax.plot(recall, precision, label=method_name, color=COLORS[i], alpha=0.8)
            ax.set_xlabel('Recall', fontsize=fontsize)
            ax.set_ylabel('Precision', fontsize=fontsize)

        ax.set_title(title, fontsize=18)

    ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='black')
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)

ax = plt.subplot(2, 3, 6)
handles = []
for color, method_name in zip(COLORS, METHODS):
    handles.append(mpatches.Patch(color=color, label=method_name))
ax.legend(handles=handles)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'roc-curves.png'), dpi=400)
