import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')

df = pd.read_csv(os.path.join(RESULTS_FOLDER, 'ot-acc.csv'), index_col=0)
df.index = [
    r'$\mathcal{D}_{9}$ protocol / $\mathcal{D}_{10}$ protocol',
    r'NovaSeq V1 chemistry ($\mathcal{D}_{6,a}$) / NovaSeq V1.5 chemistry ($\mathcal{D}_{6,b}$)',
    r'TruSeq Nano kit ($\mathcal{D}_{1,a}$) / Kapa HyperPrep kit ($\mathcal{D}_{1,b}$)',
    r'IDT indexes ($\mathcal{D}_{2,a}$) / Kapa dual indexes ($\mathcal{D}_{2,b}$)',
    r'HiSeq 2000 ($\mathcal{D}_{3,a}$) / NovaSeq ($\mathcal{D}_{3,b}$)',
    r'HiSeq 2500 ($\mathcal{D}_{4,a}$) / NovaSeq ($\mathcal{D}_{4,b}$)',
    r'HiSeq 4000 ($\mathcal{D}_{5,a}$) / NovaSeq ($\mathcal{D}_{5,b}$)',
]

plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
seaborn.heatmap(
    df, annot=True, fmt='.3f', ax=ax, square=True,
    cbar=False, linewidths=2, linecolor='white'
)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'ot-acc-heatmap.png'), dpi=400)
plt.show()