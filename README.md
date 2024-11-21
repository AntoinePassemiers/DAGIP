# DAGIP: A bias correction algorithm for cell-free DNA data

Given two groups of matched (preferentially paired) samples sequenced under different protocols, the tool explicitly learns the bias using a neural network. The approach builds on Optimal Transport theory, and exploits sample-to-sample similarities to define how to perform bias correction.

Documentations are available at [jorisvermeeschlab.github.io/DAGIP/](https://jorisvermeeschlab.github.io/DAGIP/)

---

## Install the tool

Install dependencies:
```bash
pip install -r requirements.txt
```

To reproduce the results of our manuscript, ``rpy2`` should be installed, as well the following R packages: ``dplyr``, ``GenomicRanges`` and ``dryclean``.

Install the package:
```bash
python setup.py install --user
```

---

## Basic usage:

```python
from dagip.core import DomainAdapter

# Let X and Y be two numpy arrays, where the rows are (coverage, methylation, fragmentomic) profiles and columns are features (e.g., DMRs, bins). Y and X have been produced under sequencing protocols 1 and 2, respectively.

# Build a model from the matched groups
model = DomainAdapter()
X_adapted = model.fit_transform(X, Y)

# X_adapted is a numpy array of same dimensions as X

# Save the model
model.save('some/location.pt')

...

# Load the model
model.load('some/location.pt')

# Perform bias correction on new independent samples from sequencing protocol 2
X_new_adapted = model.transform(X_new)
```

For more advanced usage, please check the documentation.

---

## Reproduce the results from the manuscript

To reproduce the results presented in our paper, first download the datasets from the FigShare repository (DOI: 10.6084/m9.figshare.24459304) and place its content in the `data` folder. The repository is available at [https://figshare.com/s/5f837a93ea2719ffcaf9](https://figshare.com/s/5f837a93ea2719ffcaf9).

Preprocess the data:
```bash
python scripts/preprocess.py
python scripts/50kb-to1mb.py
python scripts/to-numpy.py
```

#### Cancer detection

```bash
python validate-cnas.py HL
python validate-cnas.py DLBCL
python validate-cnas.py MM
python validate-cnas.py OV
python validate-fragmentomics-multimodal.py
```

#### Cross-validation with paired samples

```bash
python identify-pairs.py OV-forward
python identify-pairs.py NIPT-chemistry
python identify-pairs.py NIPT-lib
python identify-pairs.py NIPT-adapter
python identify-pairs.py NIPT-hs2000
python identify-pairs.py NIPT-hs2500
python identify-pairs.py NIPT-hs4000
```
