#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午9:09
from utils.utils import *

dataset_path = r'data/dataset.txt'


def load_dataset(path=None):
    features, labels, clicks = [], [], []
    if path is None:
        path = dataset_path
    with open(path, 'r') as file:
        for line in file:
            features_l, labels_l, clicks_l = line.split('\t')
            features.append([float(x) for x in features_l.split(',')])
            labels.append([float(x) for x in labels_l.split(',')])
            clicks.append(int(clicks_l))
    features, labels, clicks = FLOAT(features), FLOAT(labels), FLOAT(clicks)

    return features, labels, clicks


if __name__ == '__main__':
    load_dataset()
