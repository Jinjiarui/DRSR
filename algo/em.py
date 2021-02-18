import sys

sys.path.append('../')

from algo.rnn import RNN
from algo.dnn import DNN
from algo.tree import TREE
from algo.replay_buffer import *
from tools.base import *
from tools.buffer import *
from tools.utility import *

import lightgbm as lgb
import tensorflow as tf
import numpy as np


def binary_sample(probabilities):
    r = np.random.random_sample(probabilities.size)
    return (r < probabilities).astype(np.int)


class EM(BaseModel):
    def __init__(self, sess, feature_space, print_interval=500,
                 llamda=1, mu=1, nu=1, gamma=1, alpha=1, beta=0.2,
                 position_dim=32, feature_dim=32,
                 l2_weight=1e-5, learning_rate=1e-2, grad_clip=5, hidden_dim=64,
                 embedding_dim=32, emb_position_dim=30, rank_len=10, memory_size=2**17,
                 name='EM', batch_size=32, model_name=None):
        super(EM, self).__init__(sess, feature_space, name, batch_size)
        # initialize model
        if model_name == 'RNN':
            self.model = RNN(sess, feature_space, l2_weight, llamda, mu, nu, gamma, learning_rate, grad_clip, alpha, beta, hidden_dim,
                position_dim, feature_dim, rank_len, memory_size, batch_size, is_pair=False, is_bias=True)
        elif model_name == 'DNN':
            self.model = DNN(sess, feature_space, l2_weight, learning_rate, grad_clip, hidden_dim, embedding_dim, emb_position_dim,
                rank_len, memory_size, batch_size, is_bias=True)
        elif model_name == 'TREE':
            self.model = TREE(sess, feature_space)
        else:
            self.model = None
            print('================== NO MODEL DEFINED ========================')

        self.rank_len = rank_len
        self.print_interval = print_interval
        self.gamma = 1.
        self.theta = np.arange(0, rank_len, dtype=np.float32)
        self.theta = 1 / self.theta
        self.relevance_pro = np.array([])
        self.observe_pro = np.array([])

    def _build_E_step(self, data_document):
        # build Estimation for TREE model
        _theta = np.take(self.theta, data_document)
        self.relevance_pro = (1 - _theta) * self.gamma / (1 - _theta * self.gamma)
        self.observe_pro = _theta * (1 - self.gamma) / (1 - _theta * self.gamma)

    def _build_E_batch_step(self, batch):
        _theta = np.take(self.theta, batch.init_list)
        self.relevance_pro = (1 - _theta) * self.gamma / (1 - _theta * self.gamma)
        self.observe_pro = _theta * (1 - self.gamma) / (1 - _theta * self.gamma)

    def _build_M_step(self, data_document, data_feature, data_label):
        # build Maximation for TREE model
        assert isinstance(self.model, TREE)
        # build TREE model
        data = self.model.build(data_feature, data_label)
        numerator = np.zeros(self.rank_len, dtype=np.float32)
        denominator = np.zeros(self.rank_len, dtype=np.float32)
        # update theta
        for _data in range(data.num_data()):
            denominator[data_document[_data]] += 1
            numerator[data_document[_data]] += data_document[_data] + (1 - data_label[_data] * self.observe_pro[_data])
        self.theta = np.divide(numerator, denominator)
        # update gamma
        pred_label = binary_sample(self.relevance_pro)
        self.model.update_label(pred_label)
        self.gamma = self.model.train(data_feature)
        return self.gamma

    def _build_M_batch_step(self, batch):
        numerator = np.zeros(self.rank_len, dtype=np.float32)
        denominator = np.zeros(self.rank_len, dtype=np.float32)
        # update theta
        for _batch in range(self.batch_size):
            denominator[batch.init_list[_batch]] += 1
            numerator[batch.init_list[_batch]] += batch.init_list[_batch] + (1 - batch.relevance[_batch] * self.observe_pro[_batch])
        self.theta = np.divide(numerator, denominator)
        # update gamma
        print('relevance_pro', np.array(self.relevance_pro).shape)
        print(self.relevance_pro)
        pred_label = binary_sample(self.relevance_pro)
        self.gamma, _, _ = self.model.train_batch()
        # push click information into batch
        # counting click times
        # _click_count = 0.
        # _index = list(range(len(pred_label)))
        # for _click, _idx in zip(pred_label, _index):
        #     if _click == 1:
        #         click_len = _idx
        #         _click_count += 1
        #         if _click_count > 1:
        #             batch_temp = batch.copy()
        #             batch_temp.append_click_len(click_len)
        #             self._model.store_train_transition(batch_temp)
        #         elif _click_count == 1:
        #             batch.append_click_len(click_len)
        # pop current batch
        batch.pop(self.batch_size)
        # append batch with pred_label
        pred_replay_buffer = dict()
        pred_replay_buffer['pred'] = Episode()
        pred_replay_buffer['pred'].append_feature(batch.feature)
        pred_replay_buffer['pred'].append_gold(batch.gold)
        pred_replay_buffer['pred'].append_relevance(batch.relevance)
        pred_replay_buffer['pred'].append_censor(batch.censor_len)
        pred_replay_buffer['pred'].append_click(pred_label)
        pred_replay_buffer['pred'].append_event(batch.event_len)
        pred_replay_buffer['pred'].append_init(batch.init_list)
        pred_replay_buffer['pred'].append_label(batch.label)
        pred_replay_buffer['pred'].append_pair(batch.pair)
        self.model.store_transition(extract_episode(pred_replay_buffer))
        return self.gamma

    def train(self, data_document, data_feature, data_label, batch):
        print('>>>>>>TRAINING EM ...')
        if isinstance(self.model, TREE):
            self._build_E_step(data_document)
            click_rate = self._build_M_step(data_document, data_feature, data_label)
        else:
            self._build_E_batch_step(batch)
            click_rate = self._build_M_batch_step(batch)
        return click_rate

    def test(self, data_feature, data_label):
        print('>>>>>>>TESTING EM ...')
        if isinstance(self.model, TREE):
            click_rate = self.model.test(data_feature, data_label)
        else:
            click_rate = self.model.test_batch()
        return click_rate

