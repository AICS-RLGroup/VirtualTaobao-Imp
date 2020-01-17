#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午8:17
import pandas as pd
import click

from WebEye.webeye import WebEyeModel
from utils.utils import *


def load_data(path):
    df = pd.read_csv(path)
    trajectory = FLOAT(df.values)
    return trajectory


@click.command()
@click.option('--dataset_path', type=click.Path('r'), default='../data/train_data_sas.csv')
# @click.option('--batch_size', type=int, default=128, help='Batch size for GAN-SD')
# @click.option('--learning_rate_generator', 'lr_g', type=float, default=0.001, help='Learning rate for Generator')
# @click.option('--learning_rate_discriminator', 'lr_d', type=float, default=0.0001,
#               help='Learning rate for Discriminator')
@click.option('--seed', type=int, default=2019, help='Random seed for reproduce')
def main(dataset_path, seed):
    """
    Train WebEye model
    """
    expert_trajectory = load_data(dataset_path)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = WebEyeModel(expert_trajectory)
    model.train()
    model.save_model()


if __name__ == '__main__':
    main()

