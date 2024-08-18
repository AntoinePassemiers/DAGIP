import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import numpy as np
import scipy.stats
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

from dagip.utils import LaTeXTable


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results')

settings = [
    'HL', 'DLBCL', #'MM',
    'OV-forward', 'OV-backward',
    'end-motif-frequencies', 'fragment-length-distributions', 'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles'
]

for setting in settings:
    table = LaTeXTable()

    print('')
    print(setting)

    for key in ['no-target', 'baseline', 'no-source', 'center-and-scale', 'kmm', 'mapping-transport', 'da']:

        filepath = os.path.join(RESULTS_FOLDER, 'supervised-learning', f'{setting}-{key}.json')
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            data = json.load(f)

        data = data[key]

        #data = data[key]['supervised-learning']
        #data = data['svm']
        #mcc = np.mean([x['mcc'] for x in data])
        #print(key, mcc)
        print(key, data['extra'])

        #table.add(key, data)

    #print('')
    #print(setting)
    #print(table)
    #print('')
