"""Run DNN model in debiased setting"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os.path as osp
import sys

sys.path.append('../')

from algo.dnn import DNN
from tools.click import ClickModel
from tools.base import *
from tools.utility import *

base_dir = osp.dirname(osp.abspath(__file__))
log_dir = osp.join(base_dir, 'log')
data_dir = osp.join(base_dir, '../data')


def running_example(algo, running_round, training_round):
    """"run run_data.py for data generation before run run_dnn.py"""
    # initial the model
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=sess)
    summary.register(['train_NDCG3', 'train_NDCG5', 'train_MAP3', 'train_MAP5', 'test_NDCG3', 'test_NDCG5', 'test_MAP3', 'test_MAP5'])
    model = DNN(sess=sess, feature_space=700, batch_size=256)
    print("==================== START LOADING =================")
    train_batch_replay_buffer, test_batch_replay_buffer, _, _ = load_replay_buffer(data_dir=data_dir)
    model.store_train_transition(extract_episode(train_batch_replay_buffer))
    model.store_test_transition(extract_episode(test_batch_replay_buffer))
    print("================= SUCCESSFUL FINISHED LOADING ============")
    print("================= SUCCESSFUL FINISHED LOADING ============")
    for iteration in range(running_round):
        print('------- ROUND: #{} --------------'.format(iteration))
        # store metric
        train_NDCG3, train_NDCG5, test_NDCG3, test_NDCG5 = [], [], [], []
        train_MAP3, train_MAP5, test_MAP3, test_MAP5 = [], [], [], []
        T = 0
        # train
        while T < training_round:
            train_batch_NDCG3, train_batch_NDCG5 = [], []
            train_batch_MAP3, train_batch_MAP5 = [], []
            train_batch_rank_list, train_batch_gold_list, train_batch_relevance_list = model.train(print_interval=50)
            for _rank_list, _gold_list, _relevance_list in zip(train_batch_rank_list, train_batch_gold_list,
                                                               train_batch_relevance_list):
                _ndcg3, _map3 = calculate_metrics(final_list=_rank_list, gold_list=_gold_list,
                                                  relevance=_relevance_list, scope_number=3)
                _ndcg5, _map5 = calculate_metrics(final_list=_rank_list, gold_list=_gold_list,
                                                  relevance=_relevance_list, scope_number=5)
                train_batch_NDCG3.append(_ndcg3)
                train_batch_NDCG5.append(_ndcg5)
                train_batch_MAP3.append(_map3)
                train_batch_MAP5.append(_map5)
            train_NDCG3.append(np.mean(train_batch_NDCG3))
            train_NDCG5.append(np.mean(train_batch_NDCG5))
            train_MAP3.append(np.mean(train_batch_MAP3))
            train_MAP5.append(np.mean(train_batch_MAP5))
            T += 1
            print('============= TRAINING =============')
            print('>>>Mean_MAP3: [{0:<.6f}] Mean_MAP5: [{1:<.6f}] Mean_NDCG3: [{2:<.6f}] Mean_NDCG5: [{3:<.6f}]'.format(
                np.mean(train_MAP3), np.mean(train_MAP5), np.mean(train_NDCG3), np.mean(train_NDCG5)))
        # test
        test_batch_NDCG3, test_batch_NDCG5 = [], []
        test_batch_MAP3, test_batch_MAP5 = [], []
        test_batch_rank_list, test_batch_gold_list, test_batch_relevance_list = model.test(print_interval=50)
        for _rank_list, _gold_list, _relevance_list in zip(test_batch_rank_list, test_batch_gold_list,
                                                           test_batch_relevance_list):
            _ndcg3, _map3 = calculate_metrics(final_list=_rank_list, gold_list=_gold_list, relevance=_relevance_list,
                                              scope_number=3)
            _ndcg5, _map5 = calculate_metrics(final_list=_rank_list, gold_list=_gold_list, relevance=_relevance_list,
                                              scope_number=5)
            test_batch_NDCG3.append(_ndcg3)
            test_batch_NDCG5.append(_ndcg5)
            test_batch_MAP3.append(_map3)
            test_batch_MAP5.append(_map5)
        test_NDCG3.append(np.mean(test_batch_NDCG3))
        test_NDCG5.append(np.mean(test_batch_NDCG5))
        test_MAP3.append(np.mean(test_batch_MAP3))
        test_MAP5.append(np.mean(test_batch_MAP5))
        T += 1
        print('============= TESTING =============')
        print('>>>Mean_MAP3: [{0:<.6f}] Mean_MAP5: [{1:<.6f}] Mean_NDCG3: [{2:<.6f}] Mean_NDCG5: [{3:<.6f}]'.format(
            np.mean(test_MAP3), np.mean(test_MAP5), np.mean(test_NDCG3), np.mean(test_NDCG5)))
        summary.write({
            'train_NDCG3': np.mean(train_NDCG3),
            'train_NDCG5': np.mean(train_NDCG5),
            'train_MAP3': np.mean(train_MAP3),
            'train_MAP5': np.mean(train_MAP5),
            'test_NDCG3': np.mean(test_NDCG3),
            'test_NDCG5': np.mean(test_NDCG5),
            'test_MAP3': np.mean(test_MAP3),
            'test_MAP5': np.mean(test_MAP5)
        }, iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='DNN', help='Algorithm Type, choices: RNN, DNN')
    parser.add_argument('-r', '--running_round', type=int, help='Running Round Limit', default=5)
    parser.add_argument('-t', '--training_round', type=int, help='Training Round Limit', default=1)
    args = parser.parse_args()

    running_example(args.algo, args.running_round, args.training_round)
