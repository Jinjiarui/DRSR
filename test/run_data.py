"""Run data generation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os.path as osp
import sys

sys.path.append('../')

from tools.click import ClickModel
from tools.base import *
from tools.utility import *


base_dir = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(base_dir, '../data')
train_dir = osp.join(data_dir, 'train')
valid_dir = osp.join(data_dir, 'valid')
test_dir = osp.join(data_dir, 'test')

# feature: batch_size, rank_len, embed_size
# gold: batch_size, rank_len # 0-9
# relevance: batch_size, rank_len # 0-4
# click: batch_size, rank_len # 0-1
# label: batch_size, 2 # [1, 0] for click [0, 1] for non-click
# init: batch_size, rank_len # 100+
# censor: batch_size, 1 # 1-10, 10 for non-censor
# event: batch_size, 1 # 0-9, -1 for non-click
# alibaba dataset: full4test train_sample and test_sample, sample4debug train_mini_sample and test_mini_sample
# yahoo dataset: full4test train and test, sample4debug train_mini and test_mini


def running_example(click_observe_pro, non_observe_pro, rank_len, embed_size, data_name, only_click):
    print('--------------- START DATA GENERATION ----------------')
    store_replay_buffer(data_dir=data_dir, train_dir=train_dir, test_dir=test_dir,
                        non_observe_pro=non_observe_pro, click_observe_pro=click_observe_pro,
                        rank_len=rank_len, embed_size=embed_size, data_name=data_name,
                        train_data_pre='train_mini', test_data_pre='test_mini', only_click=only_click)
    print('--------------- SUCCESSFUL FINISHED DATA GENERATION ----------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--click_observe_pro', type=float, help='observe pro after click', default=0.99)
    parser.add_argument('-n', '--non_observe_pro', type=float, help='observe pro after non-click', default=0.99)
    parser.add_argument('-l', '--rank_len', type=int, help='maximum for rank cut', default=10)
    parser.add_argument('-s', '--embed_size', type=int, help='embedding size of feature engineering', default=700)
    parser.add_argument('-d', '--data_name', type=str, help='name for choosing data set: Yahoo or Alibaba', default='Yahoo')
    parser.add_argument('-o', '--only_click', type=bool, help='generate data only based on click data', default=True)
    args = parser.parse_args()

    running_example(args.click_observe_pro, args.non_observe_pro, args.rank_len, args.embed_size, args.data_name, args.only_click)
