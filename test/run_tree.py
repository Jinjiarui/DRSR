"""Run LamdaMART in debiased and biased setting"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os.path as osp
import sys

sys.path.append('../')

from algo.tree import TREE
from tools.click import ClickModel
from tools.base import *
from tools.utility import *

base_dir = osp.dirname(osp.abspath(__file__))
log_dir = osp.join(base_dir, 'log')
data_dir = osp.join(base_dir, '../data')

def operate_model_paras(paras):
    assert isinstance(paras, dict)
    return TREE(learning_rate=paras['learning_rate'], num_leaves=paras['num_leaves'])


def running_example(algo, running_round, training_round, model_paras, is_bias, is_truth):
    """run run_data.py for data generation before run run_tree.py"""
    model = operate_model_paras(paras=model_paras)
    summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=None)
    summary.register(['train_NDCG3', 'train_NDCG5', 'train_MAP3', 'train_MAP5', 'test_NDCG3', 'test_NDCG5', 'test_MAP3', 'test_MAP5'])
    print('================================== YOU ARE IN TREE MODEL ======================')
    print("==================== START LOADING =================")
    train_data_feature_gbm_file = open(data_dir + '/' + 'data_train_feature_gbm', 'rb')
    train_data_label = pickle.load(train_data_feature_gbm_file)
    train_data_feature = pickle.load(train_data_feature_gbm_file)
    train_data_feature_gbm_file.close()
    data_train_document_gbm_file = open(data_dir + '/' + 'data_train_document_gbm', 'rb')
    train_data_document = pickle.load(data_train_document_gbm_file)
    data_train_document_gbm_file.close()
    data_train_gold_gbm_file = open(data_dir + '/' + 'data_train_gold_gbm', 'rb')
    train_data_gold = pickle.load(data_train_gold_gbm_file)
    data_train_gold_gbm_file.close()
    data_train_relevance_gbm_file = open(data_dir + '/' + 'data_train_relevance_gbm', 'rb')
    train_data_relevance = pickle.load(data_train_relevance_gbm_file)
    data_train_relevance_gbm_file.close()
    data_test_feature_gbm_file = open(data_dir + '/' + 'data_test_feature_gbm', 'rb')
    test_data_label = pickle.load(data_test_feature_gbm_file)
    test_data_feature = pickle.load(data_test_feature_gbm_file)
    data_test_feature_gbm_file.close()
    data_test_document_gbm_file = open(data_dir + '/' + 'data_test_document_gbm', 'rb')
    test_data_document = pickle.load(data_test_document_gbm_file)
    data_test_document_gbm_file.close()
    data_test_gold_gbm_file = open(data_dir + '/' + 'data_test_gold_gbm', 'rb')
    test_data_gold = pickle.load(data_test_gold_gbm_file)
    data_test_gold_gbm_file.close()
    data_test_relevance_gbm_file = open(data_dir + '/' + 'data_test_relevance_gbm', 'rb')
    test_data_relevance = pickle.load(data_test_relevance_gbm_file)
    data_test_relevance_gbm_file.close()
    print("================= SUCCESSFUL FINISHED LOADING ============")

    # build TREE
    if is_bias:
        model.build(train_data_feature, train_data_label)
    if is_truth:
        model.build(train_data_feature, train_data_relevance)
    for iteration in range(running_round):
        print('------- ROUND: #{} --------------'.format(iteration))
        # store metric
        train_NDCG3, train_NDCG5, test_NDCG3, test_NDCG5 = [], [], [], []
        train_MAP3, train_MAP5, test_MAP3, test_MAP5 = [], [], [], []
        T = 0
        # train
        while T < training_round:
            train_click_list = model.train(train_data_feature)
            print(train_click_list)
            _ndcg3, _map3 = calculate_metrics_gbm(click=train_click_list, document=train_data_document, gold=train_data_gold, relevance=train_data_relevance, scope_number=3)
            _ndcg5, _map5 = calculate_metrics_gbm(click=train_click_list, document=train_data_document, gold=train_data_gold, relevance=train_data_relevance, scope_number=5)
            train_NDCG3.append(_ndcg3)
            train_NDCG5.append(_ndcg5)
            train_MAP3.append(_map3)
            train_MAP5.append(_map5)
            T += 1
            print('============= TRAINING =============')
            print('>>>Mean_MAP3: [{0:<.6f}] Mean_MAP5: [{1:<.6f}] Mean_NDCG3: [{2:<.6f}] Mean_NDCG5: [{3:<.6f}]'.format(
            np.mean(train_MAP3), np.mean(train_MAP5), np.mean(train_NDCG3), np.mean(train_NDCG5)))
        # test
        test_click_list = model.test(test_data_feature, test_data_label)
        _ndcg3, _map3 = calculate_metrics_gbm(click=test_click_list, document=test_data_document, gold=test_data_gold, relevance=test_data_relevance, scope_number=3)
        _ndcg5, _map5 = calculate_metrics_gbm(click=test_click_list, document=test_data_document, gold=test_data_gold, relevance=test_data_relevance, scope_number=3)
        test_NDCG3.append(_ndcg3)
        test_NDCG5.append(_ndcg5)
        test_MAP3.append(_map3)
        test_MAP5.append(_map5)
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
    model_paras['learning_rate'] = 0.5
    model_paras['num_leaves'] = 3


    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='TREE', help='Algorithm Type, choices: TREE')
    parser.add_argument('-r', '--running_round', type=int, help='Running Round Limit', default=1)
    parser.add_argument('-t', '--training_round', type=int, help='Training Round Limit', default=1)
    parser.add_argument('-d', '--model_paras', type=dict, help='Operating Model paras', default=model_paras)
    parser.add_argument('-b', '--is_bias', type=bool, help='Switch for bias', default=True)
    parser.add_argument('-i', '--is_truth', type=bool, help='Switch for truth', default=False)
    args = parser.parse_args()

    running_example(args.algo, args.running_round, args.training_round, args.model_paras, args.is_bias, args.is_truth)
