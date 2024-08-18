import os
import tqdm
import collections
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))

for domain in ['D11', 'D12']:
    for folder in os.listdir(os.path.join(ROOT, domain)):
        if os.path.isdir(os.path.join(ROOT, domain, folder)):
            for subfolder in ['long-fragment-ratio-profiles-50kb', 'nucleosome-positioning-score-profiles-50kb']:
                for filename in tqdm.tqdm(os.listdir(os.path.join(ROOT, domain, folder, subfolder))):
                    filepath = os.path.join(ROOT, domain, folder, subfolder, filename)
                    df = pd.read_csv(filepath)
                    regions = collections.OrderedDict()
                    scores = df['Ratio'] if (subfolder == 'long-fragment-ratio-profiles-50kb') else df['Average score']
                    for chr_id, start, end, count, ratio in zip(df['Chromosome'], df['Start'], df['End'], df['Count'], scores):
                        start, end = int(start), int(end)
                        start = start // 1000000
                        if (chr_id, start) not in regions:
                            regions[(chr_id, start)] = [0, 0.0]
                        regions[(chr_id, start)][0] += count
                        regions[(chr_id, start)][1] += ratio * count

                    new_subfolder = subfolder.replace('-50kb', '')
                    os.makedirs(os.path.join(ROOT, domain, folder, new_subfolder), exist_ok=True)
                    with open(os.path.join(ROOT, domain, folder, new_subfolder, filename), 'w') as f:
                        if new_subfolder == 'nucleosome-positioning-score-profiles':
                            f.write('Chromosome,Start,End,Count,Average score\n')
                        else:
                            f.write('Chromosome,Start,End,Count,Ratio\n')
                        for key in regions.keys():
                            chr_id, start = key
                            count = regions[key][0]
                            if count == 0:
                                count = 1
                            score = regions[key][1] / count
                            f.write(f'{chr_id},{start * 1000000 + 1},{start * 1000000 + 1000001},{count},{score:.3f}\n')
