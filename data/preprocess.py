import os
import math
import gzip
import tqdm
import random

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DATA_FOLDER = os.path.join(ROOT, 'preprocessed')


# Load list of blacklisted regions
df = pd.read_csv(os.path.join(ROOT, 'blacklisted-10kb-bins.csv'))
is_blacklisted = df['IS_BLACKLISTED'].to_numpy().astype(bool)
starts = df['START'].to_numpy()
ends = df['END'].to_numpy()


for folder in ['D1a', 'D1b', 'D2a', 'D2b', 'D3a', 'D3b', 'D4a', 'D4b', 'D5a', 'D5b', 'D6a', 'D6b', 'D7', 'D8', 'D9', 'D10']:
    for sub_folder in tqdm.tqdm(os.listdir(os.path.join(ROOT, folder, 'gc-data')), desc=folder):
        sub_folder_relative_path = os.path.join(folder, 'gc-data', sub_folder, 'gipseq.counts')
        if folder in {'D1a', 'D1b', 'D2a', 'D2b', 'D3a', 'D3b', 'D4a', 'D4b', 'D5a', 'D5b', 'D6a', 'D6b'}:
            sub_folder_relative_path = os.path.join(sub_folder_relative_path, 'w100.CM_N')
        sub_folder_path = os.path.join(ROOT, sub_folder_relative_path)
        if not os.path.isdir(sub_folder_path):
            continue
        for filename in os.listdir(sub_folder_path):
            if not filename.endswith('.tsv.gz'):
                continue
            filepath = os.path.join(sub_folder_path, filename)
            with gzip.open(filepath, 'rt') as f:
                df = pd.read_csv(f, sep='\t')
                df = df[df['CHR'] != 'chrM']
                chr_names = df['CHR'].to_numpy()
                values = df['MEAN'].to_numpy()

                assert not np.any(np.isnan(values))
                values[is_blacklisted] = np.nan

                profile, chr_names, starts, ends = [], [], [], []
                for chr_id in list(range(1, 23)):
                    chr_name = f'chr{chr_id}'
                    chr_names.append(chr_name)
                    mask = (df['CHR'] == chr_name)
                    chromosome_10kb = values[mask]
                    n_1mb_bins = int(math.ceil(len(chromosome_10kb) / 100.0))
                    for i in range(n_1mb_bins):
                        window = chromosome_10kb[i*100:(i+1)*100]
                        if (len(window) == 0) or np.all(np.isnan(window)):
                            value = np.nan
                        else:
                            value = np.nanmean(window)
                        profile.append(value)
                        chr_names.append(chr_name)
                        starts.append(i * 1000000 + 1)
                        ends.append((i + 1) * 1000000)
                profile = np.asarray(profile)
                profile = profile / np.median(profile[~np.isnan(profile)])
                profile[np.isnan(profile)] = -1

                out_filename = filename.split('.')[0] + '.csv'
                out_folder = os.path.join(ROOT, 'preprocessed', folder)
                os.makedirs(out_folder, exist_ok=True)
                out_filepath = os.path.join(out_folder, out_filename)
                with open(out_filepath, 'w') as f:
                    f.write('CHR,START,END,MEAN\n')
                    for value, chr_name, start, end in zip(profile, chr_names, starts, ends):
                        value = f'{value:.6f}' if (value >= 0) else '-1'
                        f.write(f'{chr_name},{start},{end},{value}\n')
