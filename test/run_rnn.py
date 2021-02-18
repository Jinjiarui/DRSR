"""Run RNN model in debiased setting"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os.path as osp
import sys

sys.path.append('../')

from algo.rnn import RNN
from tools.click import ClickModel
from tools.base import *
from tools.utility import *

base_dir = osp.dirname(osp.abspath(__file__))
log_dir = osp.join(base_dir, 'log')
data_dir = osp.join(base_dir, '../data')

def operate_model_paras(sess, is_pair, paras, embed_size):
    assert isinstance(paras, dict)
    return RNN(sess=sess, feature_space=embed_size, l2_weight=paras['l2_weight'],
               llamda=paras['llamda'], mu=paras['mu'], nu=paras['nu'], gamma=paras['gamma'],
               learning_rate=paras['learning_rate'], grad_clip=paras['grad_clip'],
               alpha=paras['alpha'], beta=paras['beta'], hidden_dim=paras['hidden_dim'],
               position_dim=paras['position_dim'], feature_dim=paras['feature_dim'],
               rank_len=paras['rank_len'], batch_size=paras['batch_size'], is_pair=is_pair,
               tf_device=paras['tf_device'], is_bias=False, is_truth=False)


def running_example(algo, running_round, is_pair, training_round, model_paras, train_data_pre, embed_size):
    """run run_data.py for data generation before run run_rnn.py"""
    # initial the model
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # model = RNN(sess=sess, feature_space=700, is_pair=is_pair)
    model = operate_model_paras(sess=sess, is_pair=is_pair, paras=model_paras, embed_size=embed_size)
    summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=sess)
    summary.register(['train_NDCG3', 'train_NDCG5', 'train_MAP3', 'train_MAP5', 'test_NDCG3', 'test_NDCG5', 'test_MAP3', 'test_MAP5'])
    print_type_str = '---------------- YOU ARE IN PAIR SETTING ----------------------' if is_pair else '---------------- YOU ARE IN POINT SETTING ----------------------'
    print(print_type_str)
    print("==================== START LOADING =================")
    train_batch_replay_buffer, test_batch_replay_buffer, train_batch_pair_replay_buffer, test_batch_pair_replay_buffer = load_replay_buffer(data_dir=data_dir, train_data_pre=train_data_pre)
    model.store_train_transition(extract_episode(train_batch_replay_buffer))
    model.store_test_transition(extract_episode(test_batch_replay_buffer))
    if is_pair:
        if train_batch_pair_replay_buffer is not None:
            model.store_train_transition(extract_episode(train_batch_pair_replay_buffer))
        test_batch_pair_replay_buffer = generate_permutation_order(test_batch_replay_buffer)
        if test_batch_pair_replay_buffer is not None:
            model.store_test_transition(extract_episode(test_batch_pair_replay_buffer))
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
            for _rank_list, _gold_list, _relevance_list in zip(train_batch_rank_list, train_batch_gold_list, train_batch_relevance_list):
                _ndcg3, _map3 = calculate_metrics(final_list=_rank_list, gold_list=_gold_list, relevance=_relevance_list, scope_number=3)
                _ndcg5, _map5 = calculate_metrics(final_list=_rank_list, gold_list=_gold_list, relevance=_relevance_list, scope_number=5)
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
    model_paras = dict()
    model_paras['rank_len'] = 10  # rank_len must correspond to rank_len in run_data.py
    model_paras['tf_device'] = '/cpu:*'  # use '/gpu:*' to run GPU
    # parameters of model
    model_paras['l2_weight'] = 1e-5
    model_paras['grad_clip'] = 5
    model_paras['batch_size'] = 5
    model_paras['hidden_dim'] = 64  # size of state in RNN
    model_paras['learning_rate'] = 1e-3
    model_paras['position_dim'] = 32  # size of embedding matrix of position embedding
    model_paras['feature_dim'] = 32
    # pair_loss = llamda * pair_loss_current + mu * pair_loss_before + nu * pair_loss_after
    model_paras['llamda'] = 1
    model_paras['mu'] = 1
    model_paras['nu'] = 1
    # loss = alpha * point_loss + bata * censor_loss
    model_paras['alpha'] = 1
    model_paras['beta'] = 0
    # loss += gamma * pair_loss
    model_paras['gamma'] = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='RNN', help='Algorithm Type, choices: RNN, DNN')
    parser.add_argument('-r', '--running_round', type=int, help='Running Round Limit', default=5)
    parser.add_argument('-b', '--is_pair', type=bool, help='Switch for pair-wise', default=False)
    parser.add_argument('-t', '--training_round', type=int, help='Training Round Limit', default=1)
    parser.add_argument('-m', '--model_paras', type=dict, help='Operating Model paras', default=model_paras)
    # train_data_pre and embed_size according to different data set
    parser.add_argument('-d', '--train_data_pre', type=str, help='Select Data Set', default='train_mini')
    parser.add_argument('-e', '--embed_size', type=int, help='Feature Space', default=700)
    args = parser.parse_args()

    running_example(args.algo, args.running_round, args.is_pair, args.training_round, args.model_paras, args.train_data_pre, args.embed_size)
