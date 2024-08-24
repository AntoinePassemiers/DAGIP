# -*- coding: utf-8 -*-
#
#  segmentation.py
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

import numpy as np
from sklearn.preprocessing import LabelEncoder


class SovRefine:

    class Segment:

        def __init__(self, identifier, start, end, label):
            self._id = identifier
            self._start = start
            self._end = end
            self._label = label

        def _overlap_score(self, other):
            ov = min(self._end, other._end) - max(self._start, other._start)
            return max(ov, 0)

        def overlaps(self, other):
            return self._overlap_score(other) > 0

        def minov(self, other):
            return self._overlap_score(other)

        def maxov(self, other):
            return max(self._end, other._end) - min(self._start, other._start)

        def allowance(self, other, len_s_r, delta_all):
            minov = self.minov(other)
            maxov = self.maxov(other)
            # delta = delta_all * (self.__len__() / float(len_s_r)) * (minov / float(maxov))
            delta = np.min([maxov - minov, minov, int(0.5 * self.__len__()), int(0.5 * len(other))])
            return delta

        def __str__(self):
            return str(self._label)

        def __repr__(self):
            return self.__str__()

        def __len__(self):
            return self._end - self._start

    def __init__(self, y_hat, y, scale=1.):

        assert len(y_hat) == len(y)
        encoder = LabelEncoder()
        encoder.fit(np.concatenate((y, y_hat), axis=0))
        y = encoder.transform(y)
        y_hat = encoder.transform(y_hat)

        self._n_states = len(encoder.classes_)
        self._scale = scale
        self._y_hat = np.asarray(y_hat, dtype=int)
        self._y = np.asarray(y, dtype=int)
        self._segments_hat = self.find_segments(self._y_hat)
        self._segments = self.find_segments(self._y)

        ratios = np.asarray([len(s) / float(len(self._y)) for s in self._segments])
        self._delta_all = scale * self._n_states / np.sum(ratios ** 2.)

    def find_segments(self, labels):
        segments = list()
        start = 0
        segment_ids = [0]
        N = len(labels)
        for i in range(1, N):
            if labels[i] != labels[i-1]:
                segment_ids.append(segment_ids[-1] + 1)
                segments.append(SovRefine.Segment(segment_ids[i-1], start, i, labels[i-1]))
                start = i
            else:
                segment_ids.append(segment_ids[-1])
        segments.append(SovRefine.Segment(segment_ids[N-1], start, N, labels[N-1]))
        return segments

    def filter_by_state(self, segments, i):
        for s in segments:
            if s._label == i:
                yield s

    def overlapping_segments(self, i):
        for s1 in self.filter_by_state(self._segments, i):
            for s2 in self.filter_by_state(self._segments_hat, i):
                if s1.overlaps(s2):
                    yield s1, s2

    def nonoverlapping_segments(self, i):
        for s1 in self.filter_by_state(self._segments, i):
            no_overlap = True
            for s2 in self.filter_by_state(self._segments_hat, i):
                if not s1.overlaps(s2):
                    no_overlap = False
                    break
            if no_overlap:
                yield s1

    def normalization_factor(self, i):
        N_i = sum([len(s1) for s1, _ in self.overlapping_segments(i)])
        N_i += sum([len(s1) for s1 in self.nonoverlapping_segments(i)])
        return N_i

    def unnormalized_sov(self, i):
        sov_i = 0.
        for s1, s2 in self.overlapping_segments(i):
            minov = s1.minov(s2)
            maxov = s1.maxov(s2)
            delta = s1.allowance(s2, len(self._y), self._delta_all)
            sov_i += (minov + delta) * float(len(s1)) / maxov
        return sov_i

    def sov(self, i):
        sov_i = self.unnormalized_sov(i)
        N_i = self.normalization_factor(i)
        if N_i > 0:
            return sov_i / float(N_i)
        else:
            return 1.

    def sov_refine(self):
        if np.all(self._y == self._y_hat):
            return 1.
        else:
            SOV_refine = 0.
            N = 0.
            for i in range(self._n_states):
                unnormalized_sov_i = self.unnormalized_sov(i)
                N_i = self.normalization_factor(i)
                SOV_refine += unnormalized_sov_i
                N += N_i
            SOV_refine /= N
            return SOV_refine
