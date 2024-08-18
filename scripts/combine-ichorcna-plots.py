import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn
from pdf2image import convert_from_path


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'supervised-learning')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
ICHORCNA_OUT_FOLDER = os.path.join(ROOT, '..', 'scripts', 'ichor-cna-results')


GC_CODES = [
    ('GC089534-ARD12', 'adnexal_sOV1681_P1'),
    ('GC095693-ARG04', 'adnexal_sTRA195_P0'),
    ('GC095705-ARB07', 'adnexal_sTRANS_278'),
    ('GC095705-ARG10', 'adnexal_sTRANS_294'),
    ('GC095699-ARE05', 'adnexal_sOV1808_P0')
]

method = 'dagip'
gc_code = 'GC095699-ARE05'
gc_code2 = 'adnexal_sOV1808_P0'
filepaths = [
    os.path.join(ICHORCNA_OUT_FOLDER, 'd1', gc_code, 'test', 'test_genomeWide.pdf'),
    os.path.join(ICHORCNA_OUT_FOLDER, 'd0', gc_code2, 'test', 'test_genomeWide.pdf'),
    os.path.join(ICHORCNA_OUT_FOLDER, 'd1-adapted', 'd0-controls', 'baseline', gc_code, 'test', 'test_genomeWide.pdf'),
    os.path.join(ICHORCNA_OUT_FOLDER, 'd1-adapted', 'd0-controls', 'centering-scaling', gc_code, 'test', 'test_genomeWide.pdf'),
    os.path.join(ICHORCNA_OUT_FOLDER, 'd1-adapted', 'd0-controls', 'mapping-transport', gc_code, 'test', 'test_genomeWide.pdf'),
    os.path.join(ICHORCNA_OUT_FOLDER, 'd1-adapted', 'd0-controls', 'dryclean', gc_code, 'test', 'test_genomeWide.pdf'),
    os.path.join(ICHORCNA_OUT_FOLDER, 'd1-adapted', 'd0-controls', 'dagip', gc_code, 'test', 'test_genomeWide.pdf'),
]
titles = [
    r'$\mathcal{D}_{10}$ sample with $\mathcal{D}_{10}$ panel of normals',
    r'$\mathcal{D}_{9}$ sample with $\mathcal{D}_{9}$ panel of normals',
    r'$\mathcal{D}_{10}$ sample with $\mathcal{D}_{9}$ panel of normals (Baseline)',
    r'Corrected $\mathcal{D}_{10}$ sample with $\mathcal{D}_{9}$ panel of normals (Centering-scaling)',
    r'Corrected $\mathcal{D}_{10}$ sample with $\mathcal{D}_{9}$ panel of normals (MappingTransport)',
    r'Corrected $\mathcal{D}_{10}$ sample with $\mathcal{D}_{9}$ panel of normals (dryclean)',
    r'Corrected $\mathcal{D}_{10}$ sample with $\mathcal{D}_{9}$ panel of normals (DAGIP)',
]

plt.figure(figsize=(16, 16))
for k, (title, filepath) in tqdm.tqdm(enumerate(zip(titles, filepaths))):
    plt.subplot(4, 2, k + 1)
    page = convert_from_path(filepath, 500)[0]
    page.save(os.path.join(FIGURES_FOLDER, 'temp.png'), 'PNG')
    img = mpimg.imread(os.path.join(FIGURES_FOLDER, 'temp.png'))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, f'ichorcna-plots-{gc_code}.png'), dpi=400)
plt.show()

print('Finished')
