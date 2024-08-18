import os
import numpy as np
import scipy.stats
import pandas as pd
import seaborn
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(FIGURES_FOLDER, exist_ok=True)


df = pd.read_csv(os.path.join(DATA_FOLDER, 'metadata.csv'))

plt.figure(figsize=(15, 5))

titles = [
    r'Coverage of the HEMA data set ($\mathcal{D}_7$)',
    r'Coverage of the OV data set ($\mathcal{D}_9$)',
    r'Coverage of the OV data set ($\mathcal{D}_{10}$)',
]

for k, (domain, title) in enumerate(zip(['D7', 'D9', 'D10'], titles)):

    ax = plt.subplot(1, 3, k + 1)

    sub_df = df[df['Domain'] == domain]
    controls = sub_df[sub_df['Category'] == 'Healthy']
    cases = sub_df[sub_df['Category'] != 'Healthy']

    xs = controls['Num-Reads'].to_numpy()
    ys = cases['Num-Reads'].to_numpy()

    print(domain, scipy.stats.ttest_ind(xs, ys))

    x, y = [], []
    for value in controls['Num-Reads'].to_numpy():
        x.append(value)
        y.append('Controls')
    for value in cases['Num-Reads'].to_numpy():
        x.append(value)
        y.append('Cases')
    data = {'Number of mapped reads': x, 'Cohort': y}

    seaborn.kdeplot(ax=ax, data=data, x='Number of mapped reads', hue='Cohort', palette=['royalblue', 'mediumvioletred'])
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.4, color='grey', linestyle='--', linewidth=0.5)
    ax.set_title(title)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'coverage.png'), dpi=300)
