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


def generate_exam_data(total_num=500):
    """generate data for testing model"""
    replay_buffer = dict()
    for _query in range(total_num):
        query_feature = [_feature for _feature in range(10)]
        query_relevance = [math.sin(_relevance) for _relevance in range(10)]
        query_click = [math.cos(_click) for _click in range(10)]
        replay_buffer[_query] = Episode()
        replay_buffer[_query].append_feature(query_feature)
        replay_buffer[_query].append_relevance(query_relevance)
        replay_buffer[_query].append_click(query_click)
        replay_buffer[_query].append_pair([0, 0])
        replay_buffer[_query].append_label([0, 0])
        replay_buffer[_query].append_event(9)
        replay_buffer[_query].append_censor(9)
    return replay_buffer


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
    print("=============== EXAMING ============")
    replay_buffer = generate_exam_data()
    model.store_train_transition(extract_episode(replay_buffer))
    for iteration in range(running_round):
        print('------- ROUND: #{} --------------'.format(iteration))
        T = 0
        while T < training_round:
            model.exam_rnn()
            T += 1

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
    parser.add_argument('-d', '--train_data_pre', type=str, help='Select Data Set', default='train_mini_sample')
    parser.add_argument('-e', '--embed_size', type=int, help='Feature Space', default=1)
    args = parser.parse_args()

    running_example(args.algo, args.running_round, args.is_pair, args.training_round, args.model_paras, args.train_data_pre, args.embed_size)
