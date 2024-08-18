# -*- coding: utf-8 -*-
#
#  base.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import time
import warnings
import collections
from abc import abstractmethod, ABCMeta
from typing import Dict, Tuple, Optional, Callable

import tracemalloc
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest

from dagip.benchmark.base import BaseMethod
from dagip.nipt.binning import ChromosomeBounds
from dagip.utils import LaTeXTable


class CrossValidation(metaclass=ABCMeta):

    PIPELINES = collections.OrderedDict({
        'svm': SVC(C=1, class_weight='balanced', probability=True),  # 1
        'rf': RandomForestClassifier(class_weight='balanced'),
        'reglog': LogisticRegression(C=1, class_weight='balanced', max_iter=10000)
    })

    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            sample_names: np.ndarray,
            target_domain: int = 0
    ):
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.d: np.ndarray = d
        self.sample_names: np.ndarray = np.asarray(sample_names, dtype=object)
        self.target_domain: int = int(target_domain)

        # Results
        self.table: LaTeXTable = LaTeXTable()

    def adapt(
            self,
            idx_train: np.ndarray,
            idx_test: np.ndarray,
            da_method: BaseMethod
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:

        assert len(set(list(idx_train)).intersection(set(list(idx_test)))) == 0

        d_train, d_test = self.d[idx_train], self.d[idx_test]
        assert len(np.unique(d_test)) == 1
        y_train, y_test = self.y[idx_train], self.y[idx_test]
        sample_names_train = self.sample_names[idx_train]

        X_train, X_test = self.X[idx_train], self.X[idx_test]

        # Shuffle training set
        six_train = np.arange(len(X_train))
        np.random.shuffle(six_train)
        X_train = X_train[six_train, :]
        y_train = y_train[six_train]
        d_train = d_train[six_train]
        sample_names_train = sample_names_train[six_train]

        # Shuffle test set
        six_test = np.arange(len(X_test))
        np.random.shuffle(six_test)
        X_test = X_test[six_test, :]
        y_test = y_test[six_test]
        d_test = d_test[six_test]

        # Keep track of resource consumption
        t0 = time.time()
        tracemalloc.start()

        # Sample-wise normalization
        X_train[d_train == 0, :] = da_method.normalize(
            X_train[d_train == 0, :],
            X_train[np.logical_and(y_train == 0, d_train == 0), :]
        )
        X_train[d_train == 1, :] = da_method.normalize(
            X_train[d_train == 1, :],
            X_train[np.logical_and(y_train == 0, d_train == 1), :]
        )
        X_test[d_test == 0, :] = da_method.normalize(
            X_test[d_test == 0, :],
            X_train[np.logical_and(y_train == 0, d_train == 0), :]
        )
        X_test[d_test == 1, :] = da_method.normalize(
            X_test[d_test == 1, :],
            X_train[np.logical_and(y_train == 0, d_train == 1), :]
        )

        # Domain adaptation
        X_train_adapted, weights_train = da_method.adapt(X_train, y_train, d_train, self.target_domain, subsample_target=0.25)

        # Compute resource consumption
        _, max_memory_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        extra_info = {}
        extra_info['computation-time'] = time.time() - t0
        extra_info['max-memory-usage'] = max_memory_usage

        # Evaluate overfitting of the DA algorithm
        extra_info.update(CrossValidation.evaluate_overfitting(
            X_train[d_train == 1],
            X_train_adapted[d_train == 1],
            X_train[d_train == 0]
        ))

        # Replace training data by its adapted version
        X_train = X_train_adapted

        # Inverse-shuffle the training set
        six_train = np.argsort(six_train)
        X_train = X_train[six_train, :]
        y_train = y_train[six_train]
        d_train = d_train[six_train]
        weights_train = weights_train[six_train]

        # Inverse-shuffle the test set
        six_test = np.argsort(six_test)
        X_test = X_test[six_test, :]
        y_test = y_test[six_test]
        d_test = d_test[six_test]

        # Remove unnecessary target samples from training set
        for label in np.unique(y_train):
            if np.any(np.logical_and(d_train != self.target_domain, y_train == label)):
                mask = ~np.logical_and(d_train == self.target_domain, y_train == label)
                six_train = six_train[mask]
                X_train = X_train[mask, :]
                y_train = y_train[mask]
                d_train = d_train[mask]
                weights_train = weights_train[mask]
            assert np.any(y_train == label)

        # Standard scaling
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        assert len(X_train) == len(y_train)
        assert len(X_train) == len(weights_train)
        assert len(X_test) == len(y_test)

        # Remove samples with weight=0
        mask = (weights_train > 0)
        X_train, y_train, weights_train = X_train[mask, :], y_train[mask], weights_train[mask]

        # Normalize weights
        weights_train = weights_train * len(weights_train) / np.sum(weights_train)

        return X_train, X_test, y_train, y_test, weights_train, extra_info

    @abstractmethod
    def validate(self, da_method: BaseMethod) -> dict:
        pass

    @staticmethod
    def best_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        scores = []
        for threshold in y_pred:
            scores.append(matthews_corrcoef(y_pred > threshold, y_true))
        return y_pred[np.argmax(scores)]

    @staticmethod
    def compute_evaluation_metrics(y_target: np.ndarray, y_pred: np.ndarray, train: bool = False) -> Dict[str, float]:

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Default cutoff on predictions (0.5)
            tn, fp, fn, tp = confusion_matrix(y_target, y_pred > 0.5).ravel()
            sensitivity = np.nan_to_num(tp / (tp + fn))
            specificity = np.nan_to_num(tn / (tn + fp))
            mcc = matthews_corrcoef(y_target, y_pred > 0.5)

            # Optimal cutoff on predictions
            threshold = CrossValidation.best_threshold(y_target, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_target, y_pred > threshold).ravel()
            sensitivity_best = np.nan_to_num(tp / (tp + fn))
            specificity_best = np.nan_to_num(tn / (tn + fp))
            mcc_best = matthews_corrcoef(y_target, y_pred > threshold)

        results = {
            'y': [int(label) for label in y_target],
            'y-pred': [float(pred) for pred in y_pred],
            'sensitivity': sensitivity,
            'specificity': specificity,
            'mcc': mcc,
            'sensitivity-best': sensitivity_best,
            'specificity-best': specificity_best,
            'mcc-best': mcc_best,
            'auroc': roc_auc_score(y_target, y_pred),
            'aupr': average_precision_score(y_target, y_pred)
        }
        if train:
            results = {key + '-train': value for key, value in results.items()}
        return results

    @staticmethod
    def evaluate_overfitting(X_source: np.ndarray, X_adapted: np.ndarray, X_target: np.ndarray) -> Dict[str, float]:

        # Try to re-identify source samples from their corrected version
        D = cdist(X_source, X_adapted)
        correct1 = np.arange(D.shape[0]) == np.argmin(D, axis=0)
        correct2 = np.arange(D.shape[0]) == np.argmin(D, axis=1)
        correct = np.logical_and(correct1, correct2)
        accuracy = np.mean(correct)

        # Compute coefficient of determination between adapted source samples and the closest samples in the target domain,
        # and use this coefficient as a proxy for overfitting.
        D = cdist(X_adapted, X_target)
        idx = np.argmin(D, axis=1)
        ss_res = np.mean(np.square(X_adapted - X_target[idx, :]))
        ss_tot = np.mean(np.square(X_adapted - np.mean(X_adapted, axis=0)[np.newaxis, :]))
        r2 = 1. - ss_res / ss_tot

        return {
            'accuracy-reidentification': accuracy,
            'r2-overfitting': r2
        }
