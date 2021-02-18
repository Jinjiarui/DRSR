"""Run EM for RNN and DNN model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os.path as osp
import sys

sys.path.append('../')

from algo.em import EM
from tools.click import ClickModel
from tools.base import *
from tools.utility import *

base_dir = osp.dirname(osp.abspath(__file__))
log_dir = osp.join(base_dir, 'log')
data_dir = osp.join(base_dir, '../data')


def operate_model_paras(sess, paras, model_name):
    assert isinstance(paras, dict)
    return EM(sess=sess, feature_space=paras['feature_space'], l2_weight=paras['l2_weight'],
               llamda=paras['llamda'], mu=paras['mu'], nu=paras['nu'], gamma=paras['gamma'],
               learning_rate=paras['learning_rate'], grad_clip=paras['grad_clip'],
               alpha=paras['alpha'], beta=paras['beta'], hidden_dim=paras['hidden_dim'],
               position_dim=paras['position_dim'], feature_dim=paras['feature_dim'],
               rank_len=paras['rank_len'], batch_size=paras['batch_size'],
               emb_position_dim=paras['emb_position_dim'], embedding_dim=paras['embedding_dim'],
               model_name=model_name)


def running_example(algo, running_round, model_name, training_round, model_paras):
    """run run_data.py for data generation before run run_em.py"""
    # initial the model
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = operate_model_paras(sess=sess, paras=model_paras, model_name=model_name)
    summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=sess)
    summary.register(['train_NDCG3', 'train_NDCG5', 'train_MAP3', 'train_MAP5', 'test_NDCG3', 'test_NDCG5', 'test_MAP3', 'test_MAP5'])
    print("==================== START LOADING =================")
    # set default value
    train_data_label, train_data_feature, train_data_document, train_data_gold = None, None, None, None
    test_data_label, test_data_feature, test_data_document, test_data_gold = None, None, None, None
    if model_name == 'RNN' or model_name == 'DNN':
        print('---------------- YOU ARE IN RNN OR DNN MODEL ----------------------')
        train_batch_replay_buffer, test_batch_replay_buffer, _, _ = load_replay_buffer(data_dir=data_dir)
        model.model.store_train_transition(extract_episode(train_batch_replay_buffer))
        model.model.store_test_transition(extract_episode(test_batch_replay_buffer))
    elif model_name == 'TREE':
        print('---------------- YOU ARE IN TREE MODEL ----------------------')
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
        test_data_feature_gbm_file = open(data_dir + '/' + 'data_test_feature_gbm', 'rb')
        test_data_label = pickle.load(test_data_feature_gbm_file)
        test_data_feature = pickle.load(test_data_feature_gbm_file)
        test_data_feature_gbm_file.close()
        data_test_document_gbm_file = open(data_dir + '/' + 'data_test_document_gbm', 'rb')
        test_data_document = pickle.load(data_test_document_gbm_file)
        data_test_document_gbm_file.close()
        data_test_gold_gbm_file = open(data_dir + '/' + 'data_train_gold_gbm', 'rb')
        test_data_gold = pickle.load(data_test_gold_gbm_file)
        data_test_gold_gbm_file.close()
        data_test_relevance_gbm_file = open(data_dir + '/' + 'data_test_relevance_gbm', 'rb')
        test_data_relevance = pickle.oad(data_test_relevance_gbm_file)
        data_test_relevance_gbm_file.close()
    else:
        print('---------------- NO MODEL IMPLEMENTATION -------------------')
    print("================= SUCCESSFUL FINISHED LOADING =======l=====")
    for iteration in range(running_round):
        print('------- ROUND: #{} --------------'.format(iteration))
        # store metric
        train_NDCG3, train_NDCG5, test_NDCG3, test_NDCG5 = [], [], [], []
        train_MAP3, train_MAP5, test_MAP3, test_MAP5 = [], [], [], []
        T = 0
        # train
        while T < training_round:
            train_batch = model.model.get_train_batch_data() if model_name == 'RNN' or model_name == 'DNN' else None
            train_rank_list = model.train(train_data_document, train_data_feature, train_data_label, train_batch)
            if train_batch is not None:
                _ndcg3, _map3 = calculate_metrics(final_list=train_rank_list, gold_list=train_batch.gold_list, relevance=train_batch.relevance, scope_number=3)
                _ndcg5, _map5 = calculate_metrics(final_list=train_rank_list, gold_list=train_batch.gold_list, relevance=train_batch.relevance, scope_number=5)
            else:
                _ndcg3, _map3 = calculate_metrics_gbm(click=train_rank_list, document=train_data_document, gold=train_data_gold, relevance=train_data_label, scope_number=3)
                _ndcg5, _map5 = calculate_metrics_gbm(click=train_rank_list, document=train_data_document, gold=train_data_gold, relevance=train_data_label, scope_number=5)
            train_NDCG3.append(_ndcg3)
            train_NDCG5.append(_ndcg5)
            train_MAP3.append(_map3)
            train_MAP5.append(_map5)
            T += 1
            print('============= TRAINING =============')
            print('>>>Mean_MAP3: [{0:<.6f}] Mean_MAP5: [{1:<.6f}] Mean_NDCG3: [{2:<.6f}] Mean_NDCG5: [{3:<.6f}]'.format(
            np.mean(train_MAP3), np.mean(train_MAP5), np.mean(train_NDCG3), np.mean(train_NDCG5)))
        # test
        test_batch = model.model.get_test_batch_data() if model_name == 'RNN' or model_name == 'DNN' else None
        test_rank_list = model.test(test_data_feature, test_data_label)
        if test_batch is not None:
            _ndcg3, _map3 = calculate_metrics(final_list=test_rank_list, gold_list=test_batch.gold_list, relevance=test_batch.relevance,
                                              scope_number=3)
            _ndcg5, _map5 = calculate_metrics(final_list=test_rank_list, gold_list=test_batch.gold_list, relevance=test_batch.relevance,
                                              scope_number=5)
        else:
            _ndcg3, _map3 = calculate_metrics_gbm(click=test_rank_list, document=train_data_document, gold=train_data_gold, relevance=train_data_label, scope_number=3)
            _ndcg5, _map5 = calculate_metrics_gbm(click=test_rank_list, document=train_data_document, gold=train_data_gold, relevance=train_data_label, scope_number=5)
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
    model_paras['rank_len'] = 10  # rank_len must correspond to rank_len in run_data.py
    model_paras['tf_device'] = '/cpu:*'  # use '/gpu:*' to run GPU
    # parameters of model
    model_paras['feature_space'] = 700
    model_paras['l2_weight'] = 1e-5
    model_paras['grad_clip'] = 5
    model_paras['batch_size'] = 5
    model_paras['hidden_dim'] = 64  # size of state in RNN
    model_paras['learning_rate'] = 1e-2
    model_paras['position_dim'] = 32  # size of embedding matrix of position embedding
    model_paras['feature_dim'] = 32
    # pair_loss = llamda * pair_loss_current + mu * pair_loss_before + nu * pair_loss_after
    model_paras['llamda'] = 1
    model_paras['mu'] = 1
    model_paras['nu'] = 1
    # loss = alpha * point_loss + bata * censor_loss
    model_paras['alpha'] = 1
    model_paras['beta'] = 0.2
    # loss += gamma * pair_loss
    model_paras['gamma'] = 1
    # paras for rnn
    model_paras['emb_position_dim'] = 32
    model_paras['embedding_dim'] = 32

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='EM', help='Algorithm Type, EM')
    parser.add_argument('-r', '--running_round', type=int, help='Running Round Limit', default=5)
    parser.add_argument('-n', '--model_name', type=str, help='Model in EM, Choice: RNN, DNN, TREE', default='TREE')
    parser.add_argument('-t', '--training_round', type=int, help='Training Round Limit', default=1)
    parser.add_argument('-d', '--model_paras', type=dict, help='Operating Model paras', default=model_paras)
    args = parser.parse_args()

    running_example(args.algo, args.running_round, args.model_name, args.training_round, args.model_paras)
