import tensorflow as tf
import numpy as np
import random
import math
import os
import sys
import time
import sys

sys.path.append('../')
from tools.base import BaseModel
from algo.replay_buffer import WorkerBuffer

class DNN(BaseModel):
    def __init__(self, sess, feature_space, l2_weight=1e-5,
                 learning_rate=0.05, grad_clip=5, hidden_dim=64,
                 embedding_dim=32, emb_position_dim=32, rank_len=10, memory_size=2**17,
                 name='DNN', batch_size=32, tf_device='/cpu:*', is_bias=False):
        super(DNN, self).__init__(sess, feature_space, name, batch_size)
        self.feature_space = feature_space
        self._lr = learning_rate
        self.hidden_dim = hidden_dim
        self._emb_position_dim = emb_position_dim  # embedding for position
        self.rank_len = rank_len
        self.grad_clip = grad_clip
        self._l2_weight = l2_weight
        self._embedding_dim = embedding_dim

        # trigger for debias model or directly using click or relevance information
        self.is_bias = is_bias

        # ================= DEFINE NETWORK ==========
        self.feature_ph = tf.placeholder(tf.float32, (None, self.rank_len, self.feature_space), name='feature')
        self.label_ph = tf.placeholder(tf.float32, (None, self.rank_len), name='label')
        self.click_ph = tf.placeholder(tf.float32, (None, self.rank_len), name='click')
        self.relevance_ph = tf.placeholder(tf.float32, (None, self.rank_len), name='relevance')

        self._train_replay = WorkerBuffer(memory_size)
        self._test_replay = WorkerBuffer(memory_size)
        self._build_network()
        self._build_train_op()

        # ========================= BUFFER =========================
        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(self._train_op)

    def _pos_embedding(self):
        position = tf.tile(tf.range(self.rank_len), [self.batch_size])
        position_emb = tf.reshape(position, [self.batch_size, self.rank_len])
        self.embedding_matrix = tf.random_normal(shape=[self.rank_len, self._embedding_dim], stddev=0.1)
        self.embedding_table = tf.Variable(self.embedding_matrix)
        position_emb = tf.nn.embedding_lookup(self.embedding_matrix, position_emb)  # batch, feature_space, embedding_dim
        # use position information
        return position_emb

    def _function_net(self, inp):
        emb = tf.layers.dense(inp, units=200, activation=tf.nn.relu)
        emb = tf.layers.dense(emb, units=80, activation=tf.nn.relu)
        emb = tf.layers.dense(emb, units=1, activation=tf.nn.relu)
        return emb

    # def _propensity_network_template(self, position):

    #     feature_emb = tf.layers.dense(feature, units=self._emb_position_dim, activation=tf.nn.relu)
    #     feature_emb = tf.reshape(feature_emb, shape=[self.batch_size, self.rank_len, self._emb_position_dim])
    #     propensity = tf.concat([feature_emb, tf.cast(position_emb, dtype=tf.float32)], axis=1)
    #     return propensity

    # def _rank_network_template(self, propensity):
    #     propensity_emb = tf.layers.dense(propensity, units=256, activation=tf.nn.relu)
    #     propensity_emb = tf.layers.dense(propensity, units=10, activation=tf.nn.relu)
    #     propensity_emb = tf.layers.dense(propensity, units=1, activation=tf.nn.relu)
    #     feature = tf.reshape(propensity_emb, shape=[self.batch_size, self.rank_len])
    #     return feature

    def _network_template(self, feature):
        with tf.variable_scope('Embedding'):
            embedding_name = tf.get_variable_scope().name
            pos_emb = self._pos_embedding()
            feature = tf.concat([feature, pos_emb], axis=-1)
        with tf.variable_scope('Propensity'):
            propensity_name = tf.get_variable_scope().name
            _propensity = self._function_net(pos_emb)
            _propensity = tf.squeeze(_propensity)
        with tf.variable_scope('Rank'):
            rank_name = tf.get_variable_scope().name
            _rank = self._function_net(feature)
            _rank = tf.squeeze(_rank)
        return _propensity, _rank, propensity_name, rank_name, embedding_name


    def _build_network(self):
        self.propensity_rank_net = tf.make_template('DNN', self._network_template)
        self.propensity_tf, self.rank_tf, self.propensity_name, self.rank_name, self.embedding_name = self.propensity_rank_net(self.feature_ph)

    def _cal_softmax_cross_entropy_loss(self, propensity, rank, label):
        """ calculate softmax cross_entropy_loss for both propensity and observation """
        # calculate propensity weight
        propensity_list = tf.split(tf.nn.softmax(propensity), self.rank_len, axis=1)
        _propensity_list = []
        for _size in range(self.rank_len):
            _propensity = propensity_list[0] / propensity_list[_size]
            _propensity_list.append(_propensity)
        propensity_weight = tf.concat(_propensity_list, axis=1)
        # label: batch, rank
        label = label * propensity_weight
        loss = -tf.log(tf.nn.softmax(rank, axis=-1) + 1e-10) * label
        loss = tf.reduce_sum(loss)
        return loss, propensity_weight

    def _build_train_op(self):
        # self.label = tf.reshape(self.label_ph, [self.batch_size * self.rank_len])
        self.rank_loss, propensity_weight = self._cal_softmax_cross_entropy_loss(self.propensity_tf, self.rank_tf, self.label_ph)
        self.propensity_loss, relevance_weight = self._cal_softmax_cross_entropy_loss(self.rank_tf, self.propensity_tf, self.label_ph)
        trainable_propensity = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.propensity_name)
        trainable_rank = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.rank_name)
        trainable_emb = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.embedding_name)
        trainable_rank += trainable_emb
        trainable_propensity += trainable_emb
        for _paras in trainable_propensity:
            self.propensity_loss += self._l2_weight * tf.nn.l2_loss(_paras)
        for _paras in trainable_rank:
            self.rank_loss += self._l2_weight * tf.nn.l2_loss(_paras)
        propensity_optimizer = tf.train.AdamOptimizer(self._lr)
        rank_optimizer = tf.train.AdamOptimizer(self._lr)
        # propensity_gradients, trainable_propensity = list(zip(*propensity_optimizer.compute_gradients(self.propensity_loss)))
        # propensity_gradients = tf.clip_by_global_norm(propensity_gradients, self.grad_clip)
        # rank_gradients, trainable_rank = list(zip(*rank_optimizer.compute_gradients(self.rank_loss)))
        # rank_gradients = tf.clip_by_global_norm(rank_gradients, self.grad_clip)
        # train_op_rank = rank_optimizer.apply_gradients(zip(rank_gradients, trainable_rank))
        # train_op_propensity = propensity_optimizer.apply_gradients(zip(propensity_gradients, trainable_propensity))
        train_op_propensity = propensity_optimizer.minimize(self.propensity_loss)
        train_op_rank = rank_optimizer.minimize(self.rank_loss)
        self._train_op = tf.group(train_op_propensity, train_op_rank)
        # build train_op for bias and truth experiment
        self.bias_rank_loss, _ = self._cal_softmax_cross_entropy_loss(self.propensity_tf, self.rank_tf, self.click_ph)
        self.bias_propensity_loss, _ = self._cal_softmax_cross_entropy_loss(self.rank_tf, self.propensity_tf, self.click_ph)
        bias_train_op_rank = tf.train.AdamOptimizer(self._lr).minimize(self.bias_rank_loss)
        bias_train_op_propensity = tf.train.AdamOptimizer(self._lr).minimize(self.bias_propensity_loss)
        self._bias_train_op = tf.group(bias_train_op_propensity, bias_train_op_rank)
        self.truth_rank_loss, _ = self._cal_softmax_cross_entropy_loss(self.propensity_tf, self.rank_tf, self.relevance_ph)
        self.truth_propensity_loss, _ = self._cal_softmax_cross_entropy_loss(self.rank_tf, self.propensity_tf, self.relevance_ph)
        truth_train_op_rank = tf.train.AdamOptimizer(self._lr).minimize(self.truth_rank_loss)
        truth_train_op_propensity = tf.train.AdamOptimizer(self._lr).minimize(self.truth_propensity_loss)
        self._truth_train_op = tf.group(truth_train_op_propensity, truth_train_op_rank)

    def store_train_transition(self, transitions):
        # random.shuffle(transitions)
        self._train_replay.append(transitions)

    def store_test_transition(self, transitions):
        # random.shuffle(transitions)
        self._test_replay.append(transitions)

    def test_dnn(self, print_interval):
        total_num = len(self._test_replay)
        batch_num = total_num // self.batch_size
        loss_record = [0.0, 0.0, 0.0]  # loss, censor_loss, point_loss
        print('total_number: {0:<4f}, batch_num:{1:4f}'.format(total_num, batch_num))
        rank_list = []
        gold_list = []
        relevance_list = []
        # record click rate
        for _iter in range(batch_num):
            batch = self._test_replay.sample(self.batch_size)
            label = np.zeros((self.batch_size, self.rank_len), dtype=np.float32)
            for i in range(self.batch_size):
                if batch.event_len[i] == -1:
                    continue
                else:
                    label[i][batch.event_len[i]] = 1
            test_feed_dict = {
                self.feature_ph: batch.feature,
                self.label_ph: label,
                # NOT USE
                self.click_ph: batch.click,
                self.relevance_ph: batch.relevance
            }
            click_rate, rank_loss, propensity_loss, _ = self.sess.run(
                [self.rank_tf, self.rank_loss, self.propensity_loss, self._train_op], feed_dict=test_feed_dict)
            print('click_rate', click_rate)
            batch_rank_list = []
            for _click in click_rate:
                _rank_list = list(range(len(_click)))
                _rank_map = zip(_rank_list, _click)
                _rank_map = sorted(_rank_map, key=lambda d: d[1], reverse=True)
                _rank_list = list(zip(*_rank_map))[0]
                batch_rank_list.append(_rank_list)
            rank_list.append(batch_rank_list)
            gold_list.append(batch.gold_list)
            relevance_list.append(batch.relevance)
            loss_record[0] += rank_loss / self.batch_size
            loss_record[1] += propensity_loss / self.batch_size
            if _iter % print_interval == 0:
                print('-- rank [{:<.6f}] propensity: [{:<.6f}]'.format(rank_loss, propensity_loss))
        return rank_list, gold_list, relevance_list, loss_record[0], loss_record[1]

    def train_dnn(self, print_interval):
        total_num = len(self._train_replay)
        batch_num = total_num // self.batch_size
        loss_record = [0.0, 0.0, 0.0]  # loss, censor_loss, point_loss
        print('total_number: {0:<4f}, batch_num:{1:4f}'.format(total_num, batch_num))
        rank_list = []
        gold_list = []
        relevance_list = []
        for _iter in range(batch_num):
            batch = self._train_replay.sample(self.batch_size)
            label = np.zeros((self.batch_size, self.rank_len), dtype=np.float32)
            for i in range(self.batch_size):
                if batch.event_len[i] == -1:
                    continue
                else:
                    label[i][batch.event_len[i]] = 1
            train_feed_dict = {
                self.feature_ph: batch.feature,
                self.label_ph: label,
                # NOT USE
                self.click_ph: batch.click,
                self.relevance_ph: batch.relevance
            }
            click_rate, rank_loss, propensity_loss, _ = self.sess.run(
                [self.rank_tf, self.rank_loss, self.propensity_loss, self._train_op], feed_dict=train_feed_dict)
            print('click_rate', click_rate)
            batch_rank_list = []
            for _click in click_rate:
                _rank_list = list(range(len(_click)))
                _rank_map = zip(_rank_list, _click)
                _rank_map = sorted(_rank_map, key=lambda d: d[1], reverse=True)
                _rank_list = list(zip(*_rank_map))[0]
                batch_rank_list.append(_rank_list)
            rank_list.append(batch_rank_list)
            gold_list.append(batch.gold_list)
            relevance_list.append(batch.relevance)
            loss_record[0] += rank_loss / self.batch_size
            loss_record[1] += propensity_loss / self.batch_size
            if _iter % print_interval == 0:
                print('-- rank [{:<.6f}] propensity: [{:<.6f}]'.format(rank_loss, propensity_loss))
        return rank_list, gold_list, relevance_list, loss_record[0], loss_record[1]

    def train_dnn_bias(self, print_interval):
        total_num = len(self._train_replay)
        batch_num = total_num // self.batch_size
        loss_record = [0.0, 0.0, 0.0, 0.0]  # bias_loss, truth_loss
        print('total_number: {0:<4f}, batch_num:{1:4f}'.format(total_num, batch_num))
        bias_rank_list, truth_rank_list = [], []
        gold_list = []
        relevance_list = []
        # record click rate
        for _iter in range(batch_num):
            batch = self._test_replay.sample(self.batch_size)
            label = np.zeros((self.batch_size, self.rank_len), dtype=np.float32)
            for i in range(self.batch_size):
                if batch.event_len[i] == -1:
                    continue
                else:
                    label[i][batch.event_len[i]] = 1
            train_feed_dict = {
                self.feature_ph: batch.feature,
                self.click_ph: batch.click,
                self.relevance_ph: batch.relevance,
                # NOT USE
                self.label_ph: label
            }
            bias_click_rate, _bias_rank_loss, _bias_propensity_loss, _ = self.sess.run(
                [self.rank_tf, self.bias_rank_loss, self.bias_propensity_loss, self._bias_train_op], feed_dict=train_feed_dict)
            truth_click_rate, _truth_rank_loss, _truth_propensity_loss, _ = self.sess.run(
                [self.rank_tf, self.truth_rank_loss, self.truth_propensity_loss, self._truth_train_op], feed_dict=train_feed_dict)
            batch_bias_rank_list, batch_truth_rank_list = [], []
            for _bias, _truth in zip(batch_bias_rank_list, batch_truth_rank_list):
                _bias_rank_list = list(range(len(_bias)))
                _bias_rank_map = zip(_bias_rank_list, _bias)
                _bias_rank_map = sorted(_bias_rank_map, key=lambda d: d[1], reverse=True)
                _bias_rank_list = list(zip(*_bias_rank_map))[0]
                batch_bias_rank_list.append(_bias_rank_list)
                _truth_rank_list = list(range(len(_truth)))
                _truth_rank_map = zip(_truth_rank_list, _truth)
                _truth_rank_map = sorted(_truth_rank_map, key=lambda d: d[1], reverse=True)
                _truth_rank_list = list(zip(*_truth_rank_map))[0]
                batch_truth_rank_list.append(_truth_rank_list)
            bias_rank_list.append(batch_bias_rank_list)
            truth_rank_list.append(batch_truth_rank_list)
            gold_list.append(batch.gold_list)
            relevance_list.append(batch.relevance)
            loss_record[0] += _bias_rank_loss / self.batch_size
            loss_record[1] += _bias_propensity_loss / self.batch_size
            loss_record[2] += _truth_rank_loss / self.batch_size
            loss_record[3] += _truth_propensity_loss / self.batch_size
            if _iter % print_interval == 0:
                print('-- batch #{:4f}]'.format(_iter))
                print('-- bias_rank [{:<.6f}] bias_propensity: [{:<.6f}]'.format(_bias_rank_loss, _bias_propensity_loss))
                print('-- truth_rank [{:<6f}] truth_propensity: [{:<.6f}]'.format(_truth_rank_loss, _truth_propensity_loss))
        return bias_rank_list, truth_rank_list, gold_list, relevance_list, loss_record[0], loss_record[1], loss_record[2], loss_record[3]

    def test_dnn_bias(self, print_interval):
        total_num = len(self._test_replay)
        batch_num = total_num // self.batch_size
        loss_record = [0.0, 0.0, 0.0, 0.0]  # bias_loss, truth_loss
        print('total_number: {0:<4f}, batch_num:{1:4f}'.format(total_num, batch_num))
        bias_rank_list, truth_rank_list = [], []
        gold_list = []
        relevance_list = []
        # record click rate
        for _iter in range(batch_num):
            batch = self._test_replay.sample(self.batch_size)
            label = np.zeros((self.batch_size, self.rank_len), dtype=np.float32)
            for i in range(self.batch_size):
                if batch.event_len[i] == -1:
                    continue
                else:
                    label[i][batch.event_len[i]] = 1
            test_feed_dict = {
                self.feature_ph: batch.feature,
                self.click_ph: batch.click,
                self.relevance_ph: batch.relevance,
                # NOT USE
                self.label_ph: label
            }
            bias_click_rate, _bias_rank_loss, _bias_propensity_loss, _ = self.sess.run(
                [self.rank_tf, self.bias_rank_loss, self.bias_propensity_loss, self._bias_train_op], feed_dict=test_feed_dict)
            truth_click_rate, _truth_rank_loss, _truth_propensity_loss, _ = self.sess.run(
                [self.rank_tf, self.truth_rank_loss, self.truth_propensity_loss, self._truth_train_op], feed_dict=test_feed_dict)
            batch_bias_rank_list, batch_truth_rank_list = [], []
            for _bias, _truth in zip(bias_click_rate, truth_click_rate):
                _bias_rank_list = list(range(len(_bias)))
                _bias_rank_map = zip(_bias_rank_list, _bias)
                _bias_rank_map = sorted(_bias_rank_map, key=lambda d: d[1], reverse=True)
                _bias_rank_list = list(zip(*_bias_rank_map))[0]
                batch_bias_rank_list.append(_bias_rank_list)
                _truth_rank_list = list(range(len(_truth)))
                _truth_rank_map = zip(_truth_rank_list, _truth)
                _truth_rank_map = sorted(_truth_rank_map, key=lambda d: d[1], reverse=True)
                _truth_rank_list = list(zip(*_truth_rank_map))[0]
                batch_truth_rank_list.append(_truth_rank_list)
            bias_rank_list.append(batch_bias_rank_list)
            truth_rank_list.append(batch_truth_rank_list)
            gold_list.append(batch.gold_list)
            relevance_list.append(batch.relevance)
            loss_record[0] += _bias_rank_loss / self.batch_size
            loss_record[1] += _bias_propensity_loss / self.batch_size
            loss_record[2] += _truth_rank_loss / self.batch_size
            loss_record[3] += _truth_propensity_loss / self.batch_size
            if _iter % print_interval == 0:
                print('-- batch #{:4f}]'.format(_iter))
                print('-- bias_rank [{:<.6f}] bias_propensity: [{:<.6f}]'.format(_bias_rank_loss, _bias_propensity_loss))
                print('-- truth_rank [{:<6f}] truth_propensity: [{:<.6f}]'.format(_truth_rank_loss, _truth_propensity_loss))
        return bias_rank_list, truth_rank_list, gold_list, relevance_list, loss_record[0], loss_record[1], loss_record[2], loss_record[3]

    def get_train_batch_data(self):
        # get batch_data for EM
        self._train_batch = self._train_replay.sample(self.batch_size)
        return self._train_batch

    def get_test_batch_data(self):
        # get batch_data for EM
        self._test_batch = self._test_replay.sample(self.batch_size)
        return self._test_batch

    def train_dnn_batch(self):
        bias_rank_list, gold_list, relevance_list = [], [], []
        # build label in batch
        label = np.zeros((self.batch_size, self.rank_len), dtype=np.float32)
        for i in range(self.batch_size):
            if self._train_batch.event_len[i] == -1:
                continue
            else:
                label[i][self._train_batch.event_len[i]] = 1
        train_feed_dict = {
            self.feature_ph: self._train_batch.feature,
            self.click_ph: self._train_batch.click,
            self.relevance_ph: self._train_batch.relevance,
            # NOT USE
            self.label_ph: label
        }
        bias_click_rate, _bias_rank_loss, _bias_propensity_loss, _ = self.sess.run(
                [self.rank_tf, self.bias_rank_loss, self.bias_propensity_loss, self._bias_train_op], feed_dict=train_feed_dict)

        # calculate final list
        batch_bias_rank_list = []
        for _bias in bias_click_rate:
            _bias_rank_list = list(range(len(_bias)))
            _bias_rank_map = zip(_bias_rank_list, _bias)
            _bias_rank_map = sorted(_bias_rank_map, key=lambda d: d[1], reverse=True)
            _bias_rank_list = list(zip(*_bias_rank_map))[0]
            batch_bias_rank_list.append(_bias_rank_list)
        bias_rank_list.append(batch_bias_rank_list)
        gold_list.append(self._train_batch.gold_list)
        relevance_list.append(self._train_batch.relevance)
        return bias_rank_list, gold_list, relevance_list, _bias_rank_loss, _bias_propensity_loss

    def test_dnn_batch(self):
        bias_rank_list, gold_list, relevance_list = [], [], []
        # build label in batch
        label = np.zeros((self.batch_size, self.rank_len), dtype=np.float32)
        for i in range(self.batch_size):
            if self._test_batch.event_len[i] == -1:
                continue
            else:
                label[i][self._test_batch.event_len[i]] = 1
        test_feed_dict = {
            self.feature_ph: self._test_batch.feature,
            self.click_ph: self._test_batch.click,
            self.relevance_ph: self._test_batch.relevance,
            # NOT USE
            self.label_ph: label
        }
        bias_click_rate, _bias_rank_loss, _bias_propensity_loss, _ = self.sess.run(
                [self.rank_tf, self.bias_rank_loss, self.bias_propensity_loss, self._bias_train_op], feed_dict=test_feed_dict)

        # calculate final list
        batch_bias_rank_list = []
        for _bias in bias_click_rate:
            _bias_rank_list = list(range(len(_bias)))
            _bias_rank_map = zip(_bias_rank_list, _bias)
            _bias_rank_map = sorted(_bias_rank_map, key=lambda d: d[1], reverse=True)
            _bias_rank_list = list(zip(*_bias_rank_map))[0]
            batch_bias_rank_list.append(_bias_rank_list)
        bias_rank_list.append(batch_bias_rank_list)
        gold_list.append(self._train_batch.gold_list)
        relevance_list.append(self._train_batch.relevance)
        return bias_rank_list, gold_list, relevance_list, _bias_rank_loss, _bias_propensity_loss

    def train_batch(self):
        # remember to get batch before training
        print('>>>>>TRAINING DNN ...')
        rank_list, gold_list, relevance_list, rank_loss, propensity_loss = self.train_dnn_batch()
        print('Mean_Batch_Rank_loss [[0:<.8f]] Mean_Batch_Propensity_loss [[1:<.8f]]'.format(rank_loss, propensity_loss))
        return rank_list, gold_list, relevance_list

    def test_batch(self):
        # remember to get batch before testing
        print('>>>>>>TESTING DNN ...')
        rank_list, gold_list, relevance_list, rank_loss, propensity_loss = self.test_dnn_batch()
        print('Mean_Batch_Rank_loss [[0:<.8f]] Mean_Batch_Propensity_loss [[1:<.8f]]'.format(rank_loss, propensity_loss))
        return rank_list, gold_list, relevance_list

    def train(self, print_interval=500):
        print('>>>>>>>TRAINING DNN ...')
        if self.is_bias:
            bias_rank_list, truth_rank_list, gold_list, relevance_list, bias_rank_loss, bias_propensity_loss, truth_rank_loss, truth_propensity_loss = self.train_dnn_bias(print_interval)
            print('Mean_Batch_Bias_Rank_Loss [{0:<.8f}] Mean_Batch_Bias_Loss [{1:<.8f}]'.format(bias_rank_loss, bias_propensity_loss))
            print('Mean_Batch_Truth_Rank_Loss [{0:<.8f}] Mean_Batch_Truth_Loss [{1:<.8f}]'.format(truth_rank_loss, truth_propensity_loss))
            return bias_rank_list, truth_rank_list, gold_list, relevance_list
        else:
            rank_list, gold_list, relevance_list, rank_loss, propensity_loss = self.train_dnn(print_interval)
            print('Mean_Batch_Rank_Loss [{0:<.8f}] Mean_Batch_Propensity_Loss [{1:<.8f}]'.format(rank_loss, propensity_loss))
            return rank_list, gold_list, relevance_list

    def test(self, print_interval=500):
        print('>>>>>>>TESTING DNN ...')
        if self.is_bias:
            bias_rank_list, truth_rank_list, gold_list, relevance_list, bias_rank_loss, bias_propensity_loss, truth_rank_loss, truth_propensity_loss = self.train_dnn_bias(print_interval)
            print('Mean_Batch_Bias_Rank_Loss [{0:<.8f}] Mean_Batch_Bias_Loss [{1:<.8f}]'.format(bias_rank_loss, bias_propensity_loss))
            print('Mean_Batch_Truth_Rank_Loss [{0:<.8f}] Mean_Batch_Truth_Loss [{1:<.8f}]'.format(truth_rank_loss, truth_propensity_loss))
            return bias_rank_list, truth_rank_list, gold_list, relevance_list
        else:
            rank_list, gold_list, relevance_list, rank_loss, propensity_loss = self.test_dnn(print_interval)
            print('Mean_Batch_Rank_Loss [{0:<.8f}] Mean_Batch_Propensity_Loss [{1:<.8f}]'.format(rank_loss, propensity_loss))
            return rank_list, gold_list, relevance_list
