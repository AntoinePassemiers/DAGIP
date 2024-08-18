import os
import tqdm
import zipfile
import gzip
import io

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = os.path.dirname(os.path.abspath(__file__))

meta_df = pd.read_csv(os.path.join(ROOT, 'metadata.csv'))
gc_code_dict = {gc_code: i for i, gc_code in enumerate(list(meta_df['ID']))}

dataset_domains = {
    'HEMA': ['D7', 'D8'],
    'OV': ['D9', 'D10'],
    'NIPT': ['D1a', 'D1b', 'D2a', 'D2b', 'D3a', 'D3b', 'D4a', 'D4b', 'D5a', 'D5b', 'D6a', 'D6b'],
}

for dataset in dataset_domains.keys():
    X, y, d, t, cancer_stages, gc_codes, groups, paired_with, num_reads, plasma_sep_delay = [], [], [], [], [], [], [], [], [], []
    for domain in dataset_domains[dataset]:
        for filename in os.listdir(os.path.join(ROOT, 'preprocessed', domain)):
            filepath = os.path.join(ROOT, 'preprocessed', domain, filename)
            df = pd.read_csv(filepath)
            X.append(df['MEAN'])

            # Identify paired samples
            gc_code = os.path.basename(filepath).split('.')[0]
            gc_codes.append(gc_code)
            i = gc_code_dict[gc_code]
            if meta_df.loc[i, 'Paired-With'] in gc_code_dict:
                paired_with.append(meta_df.loc[i, 'Paired-With'])
            else:
                paired_with.append('')

            # Determine category
            label = meta_df.loc[i, 'Category']
            if 'OV' in label:
                y.append('OV')
            elif label == 'Hodgkin lymphoma':
                y.append('HL')
            elif label == 'Diffuse large B-cell lymphoma':
                y.append('DLBCL')
            elif label == 'Multiple myeloma':
                y.append('MM')
            else:
                y.append('Healthy')

            # Determine domain
            d.append(domain)

            # Determine cancer stage
            cancer_stages.append(meta_df.loc[i, 'CancerStage'])

            # Determine group
            groups.append(meta_df.loc[i, 'Group'])

            # Determine confounders
            t.append(str(meta_df.loc[i, 'Lib-Prep']) + '-' + str(meta_df.loc[i, 'Sequencer']))
            plasma_sep_delay.append(meta_df.loc[i, 'Plasma-Separation-Delay'])

            # Determine number of reads
            num_reads.append(meta_df.loc[i, 'Num-Reads'])

    X = np.asarray(X)
    y = np.asarray(y, dtype=str)
    d = np.asarray(d, dtype=str)
    t = np.asarray(t, dtype=str)
    gc_codes = np.asarray(gc_codes, dtype=str)
    paired_with = np.asarray(paired_with, dtype=str)
    cancer_stages = np.asarray(cancer_stages, dtype=str)
    groups = np.asarray(groups, dtype=int)
    num_reads = np.asarray(num_reads, dtype=int)
    plasma_sep_delay = np.asarray(plasma_sep_delay)
    os.makedirs(os.path.join(ROOT, 'numpy'), exist_ok=True)
    np.savez(
        os.path.join(ROOT, 'numpy', f'{dataset}.npz'),
        X=X, y=y, d=d, t=t, gc_codes=gc_codes, groups=groups, paired_with=paired_with,
        num_reads=num_reads, plasma_sep_delay=plasma_sep_delay, cancer_stages=cancer_stages
    )
