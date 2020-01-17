#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/16 下午5:16
from typing import List

import torch
from torch.distributions import OneHotCategorical


class MultiOneHotCategorical(OneHotCategorical):
    """
        customized distribution to deal with multiple one-hot data
        Example::

        >>> m = MultiOneHotCategorical(torch.tensor([ 0.25, 0.25, 0.5, 0.2, 0.3, 0.3, 0.2]), [3, 4])
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 0.,  0.,  0.,  1.])

    """

    def __init__(self, probs: torch.Tensor, sections: List[int]):
        self._sections = sections
        self._dists = [OneHotCategorical(x) for x in torch.split(probs, sections, dim=-1)]

    def sample(self, sample_shape=torch.Size()):
        """
        concat sample from each one-hot custom together
        :param sample_shape:
        :return: [sample_dist1, sample_dist2, ...]
        """
        res = torch.cat([dist.sample() for dist in self._dists], dim=-1)
        return res

    def log_prob(self, value):
        values = torch.split(value, self._sections, dim=-1)
        log_probs = [dist.log_prob(v) for dist, v in zip(self._dists, values)]
        print(log_probs)

    def entropy(self):
        pass
