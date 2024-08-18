# -*- coding: utf-8 -*-
#
#  k_fold.py
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

from typing import Optional, Tuple, List, Callable
import traceback

import numpy as np
import statsmodels.stats.api as sms
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit

from dagip.benchmark.base import BaseMethod
from dagip.validation.base import CrossValidation


class KFoldValidation(CrossValidation):

    def __init__(
            self,
            *args,
            n_splits: int = 5,
            n_repeats: int = 10,
            average_results: bool = True,
            groups: Optional[np.ndarray] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_splits: int = int(n_splits)
        self.n_repeats: int = int(n_repeats)
        self.average_results: bool = average_results
        self.groups: Optional[np.ndarray] = groups
        self.splits: List[Tuple[np.ndarray, np.ndarray]] = self.make_splits()

    def make_splits(self, random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        idx = np.where(self.d == self.target_domain)[0]
        if self.groups is not None:
            kf = GroupShuffleSplit(n_splits=self.n_splits, random_state=random_state)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
        splits = []
        for idx_train, idx_test in kf.split(self.X[idx, :], self.y[idx], groups=(None if (self.groups is None) else self.groups[idx])):
            idx_train = idx[idx_train]
            idx_test = idx[idx_test]

            mask_train = np.zeros(len(self.d), dtype=bool)
            mask_test = np.zeros(len(self.d), dtype=bool)
            mask_train[idx_train] = True
            mask_test[idx_test] = True

            # Any sample that is not part of the target domain goes to the training set.
            mask_train[self.d != self.target_domain] = True
            mask_test[self.d != self.target_domain] = False

            # Ensure no overlap between training and test set
            mask_test[mask_train] = False

            # Split data into training/held-out sets
            idx_train = np.where(mask_train)[0]
            idx_test = np.where(mask_test)[0]

            assert len(set(list(idx_train)).intersection(set(list(idx_test)))) == 0
            assert not np.any(self.d[idx_test] != self.target_domain)

            splits.append((idx_train, idx_test))

        return splits

    def validate_by_concatenating(self, da_method: BaseMethod):

        results_ = {}
        for model_name in CrossValidation.PIPELINES.keys():
            results_[model_name] = []
        extra_infos = []

        for repeat_id in range(self.n_repeats):

            y_target_train = []
            y_target = []
            y_pred_train = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}
            y_pred = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}

            self.splits = self.make_splits(random_state=(repeat_id + 17))
            for idx_train, idx_test in self.splits:

                X_train, X_test, y_train, y_test, weights_train, extra_info = self.adapt(idx_train, idx_test, da_method)
                extra_infos.append(extra_info)

                y_target_train.append(y_train)
                y_target.append(y_test)

                for model_name, pipeline in CrossValidation.PIPELINES.items():

                    # Train supervised model
                    pipeline.fit(X_train, y_train, sample_weight=weights_train)

                    # Predict on training set
                    y_pred_train[model_name].append(pipeline.predict_proba(X_train)[:, 1])

                    # Predict on the held-out samples
                    y_pred[model_name].append(pipeline.predict_proba(X_test)[:, 1])

            y_target_train = np.squeeze(np.concatenate(y_target_train, axis=0))
            y_target = np.squeeze(np.concatenate(y_target, axis=0))
            for model_name in CrossValidation.PIPELINES.keys():
                y_pred_train[model_name] = np.squeeze(np.concatenate(y_pred_train[model_name], axis=0))
                y_pred[model_name] = np.squeeze(np.concatenate(y_pred[model_name], axis=0))
                results_[model_name].append(CrossValidation.compute_evaluation_metrics(y_target, y_pred[model_name]))
                results_[model_name][-1].update(CrossValidation.compute_evaluation_metrics(y_target_train, y_pred_train[model_name], train=True))

        """
        extra_info = {}
        if len(extra_infos) > 0:
            for key in extra_infos[0].keys():
                extra_info[key] = np.mean([x[key] for x in extra_infos])

        results = {}
        for model_name in CrossValidation.PIPELINES.keys():
            results[model_name] = {}
            for key in results_[model_name][0].keys():
                values = [float(results_[model_name][i][key]) for i in range(len(results_[model_name]))]
                results[model_name][key] = values
                results[model_name][key + '-mean'] = float(np.mean(values))
                results[model_name][key + '-confint'] = sms.DescrStatsW(values).tconfint_mean()
            #print(f'Validation results for "{model_name}": {results[model_name]}')
        """

        results = results_
        self.table.add(da_method.name(), results)

        return {
            'extra': extra_info,
            'supervised-learning': results,
        }

    def validate_by_averaging(self, da_method: BaseMethod):

        results_ = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}
        extra_infos = []

        for repeat_id in range(self.n_repeats):
            
            self.splits = self.make_splits(random_state=(repeat_id + 17))
            for split_id, (idx_train, idx_test) in enumerate(self.splits):

                try:
                    X_train, X_test, y_train, y_test, weights_train, extra_info = self.adapt(idx_train, idx_test, da_method)
                    extra_infos.append(extra_info)

                    for model_name, pipeline in CrossValidation.PIPELINES.items():

                        # Train supervised model
                        pipeline.fit(X_train, y_train, sample_weight=weights_train)

                        # Predict on the held-out sample
                        y_pred = pipeline.predict_proba(X_test)[:, 1]

                        results_[model_name].append(CrossValidation.compute_evaluation_metrics(y_test, y_pred))
                except:
                    print(traceback.format_exc())
                    for model_name, pipeline in CrossValidation.PIPELINES.items():
                        results_[model_name].append({})

        extra_info = {}
        if len(extra_infos) > 0:
            for key in extra_infos[0].keys():
                extra_info[key] = np.mean([x[key] for x in extra_infos])

        """
        results = {}
        for model_name in CrossValidation.PIPELINES.keys():
            results[model_name] = {}
            if len(results_[model_name]) > 0:
                for key in results_[model_name][0].keys():
                    values = [float(results_[model_name][i][key]) for i in range(len(results_[model_name]))]
                    results[model_name][key] = values
                    #print(sms.DescrStatsW(values).tconfint_mean())
                    results[model_name][key + '-mean'] = float(np.mean(values))
                    results[model_name][key + 'confint'] = sms.DescrStatsW(values).tconfint_mean()
                #print(f'Validation results for "{model_name}": {results[model_name]}')
        """
        results = results_
        self.table.add(da_method.name(), results)

        return {
            'supervised-learning': results,
            'extra': extra_info
        }

    def validate(self, da_method: BaseMethod):
        if self.average_results:
            return self.validate_by_averaging(da_method)
        else:
            return self.validate_by_concatenating(da_method)
