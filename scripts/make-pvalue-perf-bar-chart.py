import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results')
OUT_FOLDER = os.path.join(ROOT, '..', 'figures')


df = pd.read_csv(os.path.join(RESULTS_FOLDER, 'pvalue-performance.csv'), header='infer')

palette = {
    'Baseline': '#a883ef',
    'Center-and-scale': '#f35ee2',
    'MappingTransport': '#f4c45a',
    'dryclean': '#f996ab',
    'DAGIP': '#b1e468'
}

df_melted = pd.melt(df, id_vars='Method', var_name='Dataset', value_name='MAE')
df_melted = df_melted.dropna()
print(df_melted)

plt.figure(figsize=(4, 8))

ax = plt.subplot(1, 1, 1)
sns.barplot(ax=ax, x='MAE', y='Dataset', hue='Method', data=df_melted, palette=palette, orient='h', legend=False)
ax.set_xlabel('Mean absolute error')
ax.set_ylabel('')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
sns.despine()

for k, p in enumerate(plt.gca().patches):
    if p.get_width() > 0.08:
        plt.gca().annotate(
            f'{p.get_width():.2f}', 
            (p.get_width(), p.get_y() + p.get_height() / 2.), 
            ha='center',
            va='center',
            xytext=(-15, -1),
            textcoords='offset points',
            color='white',
            weight='bold',
            fontsize=7,
        )
    elif p.get_width() > 0:
        plt.gca().annotate(
            f'{p.get_width():.2f}', 
            (p.get_width(), p.get_y() + p.get_height() / 2.), 
            ha='center',
            va='center',
            xytext=(15, -1),
            textcoords='offset points',
            color=p.get_facecolor(),
            weight='bold',
            fontsize=7,
        )

plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, f'qq-perf.png'), dpi=400)
plt.show()