import tensorflow as tf
import numpy as np
import random
import sys
from tqdm import tqdm

sys.path.append('../')

from algo.replay_buffer import WorkerBuffer
from tools.base import BaseModel

class RNN(BaseModel):
    def __init__(self, sess, feature_space, l2_weight=1e-5, llamda=1, mu=1, nu=1, gamma=1,
                 learning_rate=1e-2, grad_clip=5, alpha=1, beta=0.2, hidden_dim=64,
                 position_dim=32, feature_dim=32, rank_len=10, memory_size=2**17,
                 name='RNN', batch_size=5, tf_device='/gpu:*', is_pair=False, is_bias=False, is_truth=False):
        super(RNN, self).__init__(sess, feature_space, name, batch_size)

        self.feature_space = feature_space
        self.hidden_dim = hidden_dim  # state dim for RNN
        self._lr = learning_rate
        self.grad_clip = grad_clip
        self._position_dim = position_dim
        self._feature_dim = feature_dim
        self.rank_len = rank_len
        self._l2_weight = l2_weight
        # weight for point-wise and pair-wise loss
        self.alpha = alpha
        self.beta = beta
        self.llamda = llamda
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        # trigger for pair-wise
        self.is_pair = is_pair
        # trigger for debias model or directly using click or relevance information
        self.is_bias = is_bias
        self.is_truth = is_truth

        # ================= DEFINE NETWORK ==========
        # with tf.device(tf_device):
        self.feature_ph = tf.placeholder(tf.float32, (None, self.rank_len, self.feature_space), name='feature')
        self.censor_ph = tf.placeholder(tf.int32, (None, ), name='censor_length')
        self.event_ph = tf.placeholder(tf.int32, (None, ), name='event_length')
        # use one-hot to label click, non-click, [0, 1] for dead/click, [1, 0] for survival/non-click
        self.label_ph = tf.placeholder(tf.float32, (None, 2), name='label')
        # use another one-hot to label pair-wise [0, 1], [1, 0] and [0, 0] for point-wise
        self.pair_ph = tf.placeholder(tf.float32, (None, 2), name='pair')
        # use relevance and click to bias-model
        self.click_ph = tf.placeholder(tf.float32, (None, self.rank_len), name='click')
        self.relevance_ph = tf.placeholder(tf.float32, (None, self.rank_len), name='relevance')

        self._train_replay = WorkerBuffer(memory_size)
        self._test_replay = WorkerBuffer(memory_size)
        self._build_network()
        self._build_train_op()

        # =================== BUFFER ====================
        self.sess.run(tf.global_variables_initializer())

    def _network_template(self, feature):
        # build embedding table for position
        seq_len = tf.maximum(self.censor_ph, self.event_ph)
        position = tf.tile(tf.range(self.rank_len), multiples=[self.batch_size])
        position_emb = tf.reshape(position, [self.batch_size, self.rank_len])
        self.embedding_matrix = tf.random_normal(shape=[self.rank_len, self._position_dim], stddev=0.1)
        self.embedding_table = tf.Variable(self.embedding_matrix)
        emb = tf.nn.embedding_lookup(self.embedding_matrix, position_emb)  # batch, rank_len, _position_dim
        # add position information
        emb = tf.concat([feature, emb], axis=-1)  # batch, rank_len, _position + _feature dim
        emb = tf.layers.dense(emb, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal())
        # preds, _ = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=self.seq_len, num_units=self.hidden_dim, dtype=tf.float32, name='RNN')
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        # emb: batch, rank_len, dim; preds: batch_size, rank_len, dim; cell: batch, hidden
        preds, _ = tf.nn.dynamic_rnn(cell, emb, dtype=tf.float32, time_major=False, sequence_length=seq_len)  # batch_size, rank_len,  dim,
        preds = tf.layers.dense(preds, units=256, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal())
        preds = tf.layers.dense(preds, units=2, activation=tf.nn.relu)
        # preds = tf.reshape(preds, [self.batch_size, self.rank_len])  # tf.squeeze
        # preds = tf.nn.sigmoid(logit)  # batch_size, rank_len
        # calculate prediction score for learning-to-rank
        # build mask
        self._seq_mask = tf.sequence_mask(self.censor_ph, maxlen=self.rank_len, dtype=tf.float32)  # batch_size, rank_len
        self.prod_score = tf.math.cumprod(preds, axis=1)  # batch_size, rank_len
        return preds

    def _build_network(self):
        self.rnn_net = tf.make_template('RNN', self._network_template)
        self.rnn_tf = self.rnn_net(self.feature_ph)
        self._build_loss_network()

    def _build_loss_network(self):
        # with tf.name_scope('cal_prod_rate'):
        #     self._cal_prod_rate()
        # with tf.name_scope('cal_censor_loss'):
        #     self._cal_censor_loss()
        with tf.name_scope('cal_point_loss'):
            self._cal_point_loss()
        # if self.is_pair:
        #     with tf.name_scope('cal_pair_loss'):
        #         self._cal_pair_loss()
        # if self.is_bias or self.is_truth:
        #     with tf.name_scope('cal_cross_entropy'):
        #         self._cal_cross_entropy_loss()

    def _cal_prod_rate(self):
        """calculate prod_rate for hazard_rate, click_rate, non_click_rate at before_click, on_click and after_click"""
        # regard rnn_tf as 1-h
        batch_rnn_rate = self.rnn_tf
        # batch_rnn_rate = tf.concat([batch_rnn_rate, tf.expand_dims(self.censor_ph, axis=-1)], axis=-1)
        # batch_rnn_rate = tf.concat([batch_rnn_rate, tf.expand_dims(self.event_ph, axis=-1)], axis=-1)
        prod_rate = tf.math.cumprod(batch_rnn_rate, axis=1)
        batch_index = tf.range(self.batch_size)
        prod_rates_before_event_two = tf.gather_nd(prod_rate, tf.stack([batch_index, (self.event_ph - 2)], axis=1))
        prod_rates_before_event_one = tf.gather_nd(prod_rate, tf.stack([batch_index, (self.event_ph - 1)], axis=1))
        prod_rates_on_event = tf.gather_nd(prod_rate, tf.stack([batch_index, (self.event_ph)], axis=1) )
        prod_rates_after_event_one = tf.gather_nd(prod_rate, tf.stack([batch_index, (self.event_ph + 1)], axis=1))
        prod_rates_after_event_two = tf.gather_nd(prod_rate, tf.stack([batch_index, (self.event_ph + 2)], axis=1))
        prod_rates_on_censor = tf.gather_nd(prod_rate, tf.stack([batch_index, (self.censor_ph)], axis=1))
        # def _cal_batch_prod_rate(_batch_rnn_rate):
        #     censor_len = tf.cast(_batch_rnn_rate[self.rank_len], dtype=tf.int32)
        #     event_len = tf.cast(_batch_rnn_rate[self.rank_len + 1], dtype=tf.int32)
        #     prod_rates_before_event_two = tf.reduce_prod(_batch_rnn_rate[0:event_len - 2])
        #     prod_rates_before_event_one = tf.reduce_prod(_batch_rnn_rate[0:event_len - 1])
        #     prod_rates_on_event = tf.reduce_prod(_batch_rnn_rate[0:event_len])
        #     prod_rates_after_event_one = tf.reduce_prod(_batch_rnn_rate[0:event_len + 1])
        #     prod_rates_after_event_two = tf.reduce_prod(_batch_rnn_rate[0:event_len + 2])
        #     prod_rate_on_censor = tf.reduce_prod(_batch_rnn_rate[0:censor_len])
        #     return tf.stack([prod_rates_before_event_two, prod_rates_before_event_one, prod_rates_on_event, prod_rates_after_event_one, prod_rates_after_event_two, prod_rate_on_censor])
        # self._prod_rate = tf.map_fn(_cal_batch_prod_rate, elems=batch_rnn_rate, name='cal_prod_rate')
        self._prod_rate = tf.stack([prod_rates_before_event_two, prod_rates_before_event_one, prod_rates_on_event, prod_rates_after_event_one, prod_rates_after_event_two, prod_rates_on_censor], axis=1)

    def _cal_censor_loss(self):
        """use label to select click and non_click data, calcualte loss for all data"""
        censor_non_click_rate = self._prod_rate[:, -1]  # prod_rate_on_censor

        censor_click_rate = 1. - censor_non_click_rate
        censor_loss = tf.transpose(tf.stack([censor_non_click_rate, censor_click_rate]), name='cal_censor_loss')
        # label: non_click: [1, 0], click: [0, 1]
        self.censor_loss = -tf.reduce_sum(self.label_ph * tf.log(tf.clip_by_value(censor_loss, 1e-10, 1.0))) / self.batch_size

    # def _cal_point_loss(self):
    #     """calculate only for click data"""
    #     # prod_rates_on_event - prod_rates_before_event_one
    #     # self.on_event_hazard_rate = tf.where(condition=self.event_ph >= 1, x=self._prod_rate[:, 1] - self._prod_rate[:, 2], y = tf.ones_like(self._prod_rate[:, 2]))
    #
    #     self.on_event_hazard_rate = tf.where(condition=self.event_ph >= 1, x=self._prod_rate[:, 1] - self._prod_rate[:, 2], y=1. - self._prod_rate[:, 2])
    #     # self.on_event_hazard_rate = tf.where(condition=tf.cast(tf.argmax(self.label_ph, axis=1), tf.bool), x=self.on_event_hazard_rate, y=tf.ones_like(self.on_event_hazard_rate))
    #     # log_minus = - tf.log(tf.add(self.on_event_hazard_rate, 1e-20))
    #     self.point_loss = tf.losses.log_loss(tf.argmax(self.label_ph, axis=1), self.on_event_hazard_rate, weights=tf.argmax(self.label_ph, axis=1))
    #
    #     # calculate number of click in batch
    #     # point_batch_size = tf.reduce_sum(self.label_ph, axis=0)[1]
    #     # self.point_loss = tf.cond(
    #     #     tf.cast(point_batch_size, tf.bool),
    #     #     lambda: tf.reduce_sum(log_minus * tf.cast(tf.argmax(self.label_ph, axis=1), tf.float32)) / point_batch_size,
    #     #     lambda: tf.constant(0, dtype=tf.float32))

    def _cal_pair_loss(self):
        # calculate Probability maximum P(click relevance GIVEN observe irrelevance)
        # on_event_hazard rate GIVEN before_event_one_non_click_rate
        # probability based on original order - current order
        on_event_hazard_rate = tf.where(condition=self.event_ph >= 1, x=self._prod_rate[:, 1] - self._prod_rate[:, 2], y=tf.ones_like(self._prod_rate[:, 2]))
        log_on_event_hazard_rate = tf.log(tf.add(on_event_hazard_rate, 1e-20))
        before_event_one_non_click_rate = tf.where(condition=self.event_ph >= 1, x=self._prod_rate[:, 1], y=tf.ones_like(self._prod_rate[:, 1])) # prod_rates_before_event_one
        log_current = tf.log(tf.add(before_event_one_non_click_rate, 1e-20))
        log_current = log_current - log_on_event_hazard_rate
        # calculate number of pair_loss_max_before in batch
        pair_current_batch_size = self.batch_size - tf.reduce_sum(self.pair_ph, axis=0)[1]
        pair_loss_current = tf.cond(
            tf.cast(pair_current_batch_size, tf.bool),
            lambda: tf.reduce_sum(log_current * tf.cast((1 - tf.argmax(self.pair_ph, axis=1)), tf.float32)) / pair_current_batch_size,
            lambda: tf.constant(0, dtype=tf.float32))
        # calculate Probability minimum P(click irrelevance GIVEN observe relevance)
        # on_event_non_click_rate GIVEN after_event_one_hazard_rate
        # probability based on re-rank order zero - before order
        on_event_non_click_rate = self._prod_rate[:, 2]
        log_on_event_non_click_rate = - tf.log(tf.add(on_event_non_click_rate, 1e-20))
        after_event_one_hazard_rate = tf.where(condition=self.censor_ph-self.event_ph >= 1, x=self._prod_rate[:, 2] - self._prod_rate[:, 3], y=1. - self._prod_rate[:, 2])
        after_event_one_hazard_rate = tf.where(condition=self.event_ph >= 1, x=after_event_one_hazard_rate, y=tf.ones_like(after_event_one_hazard_rate))
        log_before = tf.log(tf.add(after_event_one_hazard_rate, 1e-20))
        log_before = log_on_event_non_click_rate - log_before
        pair_before_batch_size = tf.reduce_sum(self.pair_ph, axis=0)[0]
        pair_loss_before = tf.cond(
            tf.cast(pair_before_batch_size, tf.bool),
            lambda: tf.reduce_sum(log_before * tf.cast(self.pair_ph[:, 0], tf.float32)) / pair_before_batch_size,
            lambda: tf.constant(0, dtype=tf.float32))
        # calculate Probability maximum P(observe uncertainly irrelevance GIVEN observe irrelevance)
        # before_event_one_non_click_rate GIVEN before_event_two_non_click_rate
        # probability based on re-rank one - after order
        before_event_two_non_click_rate = self._prod_rate[:, 0]
        log_after = tf.log(tf.add(before_event_two_non_click_rate, 1e-20))
        log_after = log_before - log_after
        pair_after_batch_size = tf.reduce_sum(self.pair_ph, axis=0)[1]
        pair_loss_after = tf.cond(
            tf.cast(pair_after_batch_size, tf.bool),
            lambda: tf.reduce_sum(log_after * tf.cast(tf.argmax(self.pair_ph, axis=1), tf.float32)) / pair_after_batch_size,
            lambda: tf.constant(0, dtype=tf.float32))
        # computer pair_loss_current for [0, 0] (point-wise order) and [0, 1] (before order)
        # computer pair_loss_before for [0, 1] (before order)
        # computer pair_loss_after for [1, 0] (after order)
        self.pair_loss = self.llamda * pair_loss_current + self.mu * pair_loss_before + self.nu * pair_loss_after

    def _cal_cross_entropy_loss(self):
        log_minus = - tf.log(tf.add(self.rnn_tf, 1e-20))
        # calculate number of click in batch
        self._bias_loss = tf.reduce_sum(log_minus * self.click_ph) / self.batch_size
        self._truth_loss = tf.reduce_sum(log_minus * self.relevance_ph) / self.batch_size

    def _cal_point_loss(self):
        logits = self.rnn_tf
        labels = tf.cast(self.click_ph, dtype=tf.int32)
        self.point_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        self.point_loss = tf.reduce_sum(self.point_loss * self._seq_mask) / self.batch_size
        self.censor_loss = tf.zeros_like(self.point_loss)

    def _build_train_op(self):
        trainable_vars = tf.trainable_variables()
        # self._loss = self.alpha * self.point_loss + self.beta * self.censor_loss
        # if self.is_pair:
        #     self._loss += self.pair_loss * self.gamma
        self._loss = self.point_loss
        l2_loss = tf.add_n([tf.nn.l2_loss(_var) for _var in trainable_vars]) * self._l2_weight
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss + l2_loss, trainable_vars), self.grad_clip)
        self._train_op = tf.train.AdamOptimizer(learning_rate=self._lr).apply_gradients(zip(grads, trainable_vars))
        # self._train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr).apply_gradients(zip(grads, trainable_vars))
        if self.is_bias:
            bias_grads, _ = tf.clip_by_global_norm(tf.gradients(self._bias_loss + l2_loss, trainable_vars), self.grad_clip)
            self._bias_train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr).apply_gradients(zip(bias_grads, trainable_vars))
        if self.is_truth:
            truth_grads, _ = tf.clip_by_global_norm(tf.gradients(self._truth_loss + l2_loss, trainable_vars), self.grad_clip)
            self._truth_train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr).apply_gradients(zip(truth_grads, trainable_vars))

    def store_train_transition(self, transitions):
        # random.shuffle(transitions)
        self._train_replay.append(transitions)

    def store_test_transition(self, transitions):
        # random.shuffle(transitions)
        self._test_replay.append(transitions)

    def test_rnn(self, print_interval):
        # print(">>>>>>> TESTING =========")
        total_num = len(self._test_replay)
        batch_num = total_num // self.batch_size
        loss_record = [0.0, 0.0, 0.0]  # loss, censor_loss, point_loss
        # record and return final, gold, relevance
        rank_list = []
        gold_list = []
        relevance_list = []
        print('total_number: {0:<4d}, batch_num:{1:4d}'.format(total_num, batch_num))
        for _iter in range(batch_num):
            batch = self._test_replay.sample(self.batch_size)
            train_feed_dict = {
                self.feature_ph: batch.feature,
                self.label_ph: batch.label,
                self.event_ph: batch.event_len,
                self.censor_ph: batch.censor_len,
                self.pair_ph: batch.pair,
                # NOT USE
                self.click_ph: batch.click,
                self.relevance_ph: batch.relevance
            }
            # print('======================================== DEBUG =============================================')
            # for _feature, _label, _event, _censor, _pair, _click, _relevance in zip(batch.feature, batch.label, batch.event_len, batch.pair, batch.click, batch.relevance):
            #     print('batch_feature', np.array(batch.feature).shape)
            #     print(_feature)
            #     print('batch_label', np.array(batch.label).shape)
            #     print(_label)
            #     print('batch_event', np.array(batch.event_len).shape)
            #     print(_event)
            #     print('batch_censor', np.array(batch.censor_len).shape)
            #     print(_censor)
            #     print('batch_pair', np.array(batch.pair).shape)
            #     print(_pair)
            #     print('batch_click', np.array(batch.click).shape)
            #     print(_click)
            #     print('batch_relevance', np.array(batch.relevance).shape)
            #     print(_relevance)
            # calculate element_loss to represent point or pair loss
            if self.is_pair:
                click_rate, censor_loss, element_loss, _loss, _ = self.sess.run([self.prod_score, self.censor_loss, self.pair_loss, self._loss, self._train_op], feed_dict=train_feed_dict)
            else:
                click_rate, censor_loss, element_loss, _loss, _ = self.sess.run([self.prod_score, self.censor_loss, self.point_loss, self._loss, self._train_op], feed_dict=train_feed_dict)
            # calculate final list
            batch_rank_list = []
            for _click, _len in zip(click_rate, batch.censor_len):
                _click = _click[:_len]
                _click = _click[-1]  # click_pro for classification lose
                _rank_list = list(range(len(_click)))
                _rank_map = zip(_rank_list, _click)
                _rank_map = sorted(_rank_map, key=lambda d: d[1], reverse=True)
                _rank_list = list(zip(*_rank_map))[0]
                batch_rank_list.append(_rank_list)
                # batch_rank_list.append(range(10))
            rank_list.append(batch_rank_list)
            gold_list.append(batch.gold_list)
            relevance_list.append(batch.relevance)
            # calculate loss
            loss_record[0] += _loss / self.batch_size
            loss_record[1] += censor_loss / self.batch_size
            loss_record[2] += element_loss / self.batch_size

            if _iter % print_interval == 0:
                print('-- batch #{:<4f} loss: [{:<.6f}]'.format(_iter, _loss))
                print('-- censor [{:<.6f}] element: [{:<.6f}]'.format(censor_loss, element_loss))
        return rank_list, gold_list, relevance_list, loss_record[0]

    def train_rnn(self, print_interval):
        # print('>>>> TRAINING ==========')
        total_num = len(self._train_replay)
        batch_num = total_num // self.batch_size
        loss_record = [0.0, 0.0, 0.0]  # loss, censor_loss, point_loss
        # record and return final, gold, relevance
        rank_list = []
        gold_list = []
        relevance_list = []
        print('total_number: {0:<4d}, batch_num:{1:<4d}'.format(total_num, batch_num))
        for _iter in range(batch_num):
            batch = self._train_replay.sample(self.batch_size)
            # print('======================= DEBUG ====================')
            # print('Feature', type(batch.feature))
            # print(np.array(batch.feature).shape)
            # print('Label', type(batch.label))
            # print(batch.label)
            # print(np.array(batch.label).shape)
            # print('Censor', type(batch.censor_len))
            # print(batch.censor_len)
            # print(np.array(batch.censor_len).shape)
            # print('Pair', type(batch.pair))
            # print(batch.pair)
            # print(np.array(batch.pair).shape)
            # print('Click', type(batch.click))
            # print(batch.click)
            # print(np.array(batch.click).shape)
            # print('Relevance', type(batch.relevance))
            # print(batch.relevance)
            # print(np.array(batch.relevance).shape)
            train_feed_dict = {
                self.feature_ph: batch.feature,
                self.label_ph: batch.label,
                self.event_ph: batch.event_len,
                self.censor_ph: batch.censor_len,
                self.pair_ph: batch.pair,
                # NOT USE
                self.click_ph: batch.click,
                self.relevance_ph: batch.relevance
            }
            # calculate element_loss to represent point or pair loss
            if self.is_pair:
                click_rate, censor_loss, element_loss, _loss, _ = self.sess.run([self.prod_score, self.censor_loss, self.pair_loss, self._loss, self._train_op], feed_dict=train_feed_dict)
            else:
                click_rate, censor_loss, element_loss, _loss, _, _rnn = self.sess.run([self.prod_score, self.censor_loss, self.point_loss, self._loss, self._train_op, self.rnn_tf], feed_dict=train_feed_dict)
            # calculate final list
            # print('RNN', _rnn)
            # print('batch.click', batch.click)
            # print('batch.relevance', batch.relevance)
            # print('batch.censor_len', batch.censor_len)
            # print('batch.event_len', batch.event_len)
            batch_rank_list = []
            for _click, _len in zip(click_rate, batch.censor_len):
                _click = _click[:_len]
                _click = _click[-1]  # click_pro for classification lose
                _rank_list = list(range(len(_click)))
                _rank_map = zip(_rank_list, _click)
                _rank_map = sorted(_rank_map, key=lambda d: d[1], reverse=True)
                _rank_list = list(zip(*_rank_map))[0]
                batch_rank_list.append(_rank_list)
            rank_list.append(batch_rank_list)
            gold_list.append(batch.gold_list)
            relevance_list.append(batch.relevance)

            # calculate loss
            loss_record[0] += _loss / self.batch_size
            loss_record[1] += censor_loss / self.batch_size
            loss_record[2] += element_loss / self.batch_size

            if _iter % print_interval == 0:
                print('-- batch #{:<4f} loss: [{:<.6f}]'.format(_iter, _loss))
                print('-- censor [{:<.6f}] element: [{:<.6f}]'.format(censor_loss, element_loss))
        return rank_list, gold_list, relevance_list, loss_record[0]

    def get_train_batch_data(self):
        # get batch_data for EM
        self._train_batch = self._train_replay.sample(self.batch_size)
        return self._train_batch

    def get_test_batch_data(self):
        # get batch_data for EM
        self._test_batch = self._test_replay.sample(self.batch_size)
        return self._test_batch

    def train_rnn_batch(self):
        bias_rank_list, gold_list, relevance_list = [], [], []
        # run _get_test_batch_data method before running this method
        train_feed_dict = {
            self.feature_ph: self._train_batch.feature,
            self.click_ph: self._train_batch.click,
            self.relevance_ph: self._train_batch.relevance,
            # NOT USE
            self.censor_ph: self._train_batch.censor_len,
            self.event_ph: self._train_batch.event_len,
            self.label_ph: self._train_batch.label,
            self.pair_ph: self._train_batch.pair}
        # calculate element_loss to represent point or pair loss
        bias_click_rate, _bias_loss, _ = self.sess.run([self.prod_score, self._bias_loss, self._bias_train_op],
                                                           feed_dict=train_feed_dict)
        # calculate final list
        batch_bias_rank_list = []
        for _bias in bias_click_rate:
            _bias_rank_list = list(range(len(_bias)))
            _bias_rank_map = zip(_bias_rank_list, _bias)
            _bias_rank_map = sorted(_bias_rank_map, key=lambda d: d[1], reverse=False)
            _bia_rank_list = list(zip(*_bias_rank_map))[0]
            batch_bias_rank_list.append(_bias_rank_list)
        bias_rank_list.append(batch_bias_rank_list)
        gold_list.append(self._train_batch.gold_list)
        relevance_list.append(self._train_batch.relevance)
        print('-- bias [{:<.6f}]'.format(_bias_loss))
        return bias_rank_list, gold_list, relevance_list, _bias_loss

    def test_rnn_batch(self):
        bias_rank_list, gold_list, relevance_list = [], [], []
        # run _get_test_batch_data method before running this method
        test_feed_dict = {
            self.feature_ph: self._test_batch.feature,
            self.click_ph: self._test_batch.click,
            self.relevance_ph: self._test_batch.relevance,
            # NOT USE
            self.censor_ph: self._test_batch.censor_len,
            self.event_ph: self._test_batch.event_len,
            self.label_ph: self._test_batch.label,
            self.pair_ph: self._test_batch.pair}
        # calculate element_loss to represent point or pair loss
        bias_click_rate, _bias_loss = self.sess.run([self.prod_score, self._bias_loss],
                                                           feed_dict=test_feed_dict)
        # calculate final list
        batch_bias_rank_list, batch_truth_rank_list = [], []
        for _bias in bias_click_rate:
            _bias_rank_list = list(range(len(_bias)))
            _bias_rank_map = zip(_bias_rank_list, _bias)
            _bias_rank_map = sorted(_bias_rank_map, key=lambda d: d[1], reverse=False)
            _bia_rank_list = list(zip(*_bias_rank_map))[0]
            batch_bias_rank_list.append(_bias_rank_list)
        bias_rank_list.append(batch_bias_rank_list)
        gold_list.append(self._train_batch.gold_list)
        relevance_list.append(self._train_batch.relevance)
        print('-- bias [{:<.6f}]'.format(_bias_loss))
        return bias_rank_list, gold_list, relevance_list, _bias_loss

    def train_rnn_bias(self, print_interval):
        # print('>>>> TRAINING ==========')
        _bias_loss, _truth_loss = 0., 0.
        total_num = len(self._train_replay)
        batch_num = total_num // self.batch_size
        loss_record = 0.0  # loss, bias_loss, truth_loss
        # record and return final, gold, relevance
        bias_rank_list, truth_rank_list = [], []
        gold_list = []
        relevance_list = []
        print('total_number: {0:<4d}, batch_num:{1:<4d}'.format(total_num, batch_num))
        for _iter in range(batch_num):
            batch = self._train_replay.sample(self.batch_size)
            print('======================= DEBUG ====================')
            print('Feature', type(batch.feature))
            print(np.array(batch.feature).shape)
            print('Label', type(batch.label))
            print(np.array(batch.label).shape)
            print('Censor', type(batch.censor_len))
            print(np.array(batch.censor_len).shape)
            print('Pair', type(batch.pair))
            print(np.array(batch.pair).shape)
            print('Click', type(batch.click))
            print(np.array(batch.click).shape)
            print('Relevance', type(batch.relevance))
            print(np.array(batch.relevance).shape)
            train_feed_dict = {
                self.feature_ph: batch.feature,
                self.click_ph: batch.click,
                self.relevance_ph: batch.relevance,
                # NOT USE
                self.censor_ph: batch.censor_len,
                self.event_ph: batch.event_len,
                self.label_ph: batch.label,
                self.pair_ph: batch.pair
            }
            # calculate element_loss to represent point or pair loss
            batch_bias_rank_list, batch_truth_rank_list = [], []
            if self.is_bias:
                bias_click_rate, _bias_loss, _ = self.sess.run([self.rnn_tf, self._bias_loss, self._bias_train_op], feed_dict=train_feed_dict)
                for _bias in bias_click_rate:
                    _bias_rank_list = list(range(len(_bias)))
                    _bias_rank_map = zip(_bias_rank_list, _bias)
                    _bias_rank_map = sorted(_bias_rank_map, key=lambda d: d[1], reverse=False)
                    _bia_rank_list = list(zip(*_bias_rank_map))[0]
                    batch_bias_rank_list.append(_bias_rank_list)
                bias_rank_list.append(batch_bias_rank_list)
            if self.is_truth:
                truth_click_rate, _truth_loss, _ = self.sess.run([self.rnn_tf, self._truth_loss, self._truth_train_op], feed_dict=train_feed_dict)
                for _truth in truth_click_rate:
                    _truth_rank_list = list(range(len(_truth)))
                    _truth_rank_map = zip(_truth_rank_list, _truth)
                    _truth_rank_map = sorted(_truth_rank_map, key=lambda d: d[1], reverse=False)
                    _truth_rank_list = list(zip(*_truth_rank_map))[0]
                    batch_truth_rank_list.append(_truth_rank_list)
            truth_rank_list.append(batch_truth_rank_list)
            # calculate final list
            gold_list.append(batch.gold_list)
            relevance_list.append(batch.relevance)

            # calculate loss
            loss_record += _bias_loss / self.batch_size if self.is_bias else _truth_loss / self.batch_size

            if _iter % print_interval == 0:
                print('-- batch #{:<4f}]'.format(_iter))
                print('-- bias [{:<.6f}] truth: [{:<.6f}]'.format(_bias_loss, _truth_loss))
        if self.is_bias:
            return bias_rank_list, gold_list, relevance_list, loss_record
        elif self.is_truth:
            return truth_rank_list, gold_list, relevance_list, loss_record
        else:
            return NotImplementedError

    def test_rnn_bias(self, print_interval):
        # print('>>>> TRAINING ==========')
        _bias_loss, _truth_loss = 0., 0.
        total_num = len(self._test_replay)
        batch_num = total_num // self.batch_size
        loss_record = 0.0  # loss, bias_loss, truth_loss
        # record and return final, gold, relevance
        bias_rank_list, truth_rank_list = [], []
        gold_list = []
        relevance_list = []
        print('total_number: {0:<4d}, batch_num:{1:<4d}'.format(total_num, batch_num))
        for _iter in range(batch_num):
            batch = self._test_replay.sample(self.batch_size)
            test_feed_dict = {
                self.feature_ph: batch.feature,
                self.click_ph: batch.click,
                self.relevance_ph: batch.relevance,
                # NOT USE
                self.label_ph: batch.label,
                self.event_ph: batch.event_len,
                self.censor_ph: batch.censor_len,
                self.pair_ph: batch.pair
            }
            # calculate element_loss to represent point or pair loss
            # calculate final list
            batch_bias_rank_list, batch_truth_rank_list = [], []
            if self.is_bias:
                bias_click_rate, _bias_loss, = self.sess.run([self.rnn_tf, self._bias_loss], feed_dict=test_feed_dict)
                for _bias in bias_click_rate:
                    _bias_rank_list = list(range(len(_bias)))
                    _bias_rank_map = zip(_bias_rank_list, _bias)
                    _bias_rank_map = sorted(_bias_rank_map, key=lambda d: d[1], reverse=False)
                    _bia_rank_list = list(zip(*_bias_rank_map))[0]
                    batch_bias_rank_list.append(_bias_rank_list)
                bias_rank_list.append(batch_bias_rank_list)
            if self.is_truth:
                truth_click_rate, _truth_loss = self.sess.run([self.rnn_tf, self._truth_loss], feed_dict=test_feed_dict)
                for _truth in truth_click_rate:
                    _truth_rank_list = list(range(len(_truth)))
                    _truth_rank_map = zip(_truth_rank_list, _truth)
                    _truth_rank_map = sorted(_truth_rank_map, key=lambda d: d[1], reverse=False)
                    _truth_rank_list = list(zip(*_truth_rank_map))[0]
                    batch_truth_rank_list.append(_truth_rank_list)
                truth_rank_list.append(batch_truth_rank_list)
            gold_list.append(batch.gold_list)
            relevance_list.append(batch.relevance)

            # calculate loss
            loss_record += _bias_loss / self.batch_size if self.is_bias else _truth_loss / self.batch_size

            if _iter % print_interval == 0:
                print('-- batch #{:<4f}]'.format(_iter))
                print('-- bias [{:<.6f}] truth: [{:<.6f}]'.format(_bias_loss, _truth_loss))
        if self.is_bias:
            return bias_rank_list, gold_list, relevance_list, loss_record
        elif self.is_truth:
            return truth_rank_list, gold_list, relevance_list, loss_record
        else:
            return NotImplementedError

    def train(self, print_interval=500):
        print('>>>>>TRAINING RNN ...')
        if self.is_bias:
            bias_rank_list, gold_list, relevance_list, bias_loss = self.train_rnn_bias(print_interval)
            print("Mean_Batch_Bias_Loss [{0:<.8f}]".format(bias_loss))
            return bias_rank_list, gold_list, relevance_list
        elif self.is_truth:
            truth_rank_list, gold_list, relevance_list, truth_loss = self.train_rnn_bias(print_interval)
            print("Mean_Batch_Truth_Loss [{0:<.8f}]".format(truth_loss))
            return truth_rank_list, gold_list, relevance_list
        else:
            rank_list, gold_list, relevance_list, loss = self.train_rnn(print_interval)
            print('Mean_Batch_Loss [{0:<.8f}]'.format(loss))
            return rank_list, gold_list, relevance_list

    def train_batch(self):
        # remember to get batch before training
        print('>>>>>TRAINING RNN ...')
        rank_list, gold_list, relevance_list, loss = self.train_rnn_batch()
        print('Mean_Batch_loss [{0:<.8f}]'.format(loss))
        return rank_list, gold_list, relevance_list

    def test_batch(self):
        # remember to get batch before testing
        print('>>>>>>TESTING RNN ...')
        rank_list, gold_list, relevance_list, loss = self.test_rnn_batch()
        print('Mean_Batch_loss [{0:<.8f}]'.format(loss))
        return rank_list, gold_list, relevance_list

    def test(self, print_interval=500):
        print('>>>>>>>TESTING RNN ...')
        if self.is_bias:
            bias_rank_list, gold_list, relevance_list, bias_loss = self.test_rnn_bias(print_interval)
            print("Mean_Batch_Bias_Loss [{0:<.8f}]".format(bias_loss))
            return bias_rank_list, gold_list, relevance_list
        elif self.is_truth:
            truth_rank_list, gold_list, relevance_list, truth_loss = self.test_rnn_bias(
                print_interval)
            print("Mean_Batch_Truth_Loss [{0:<.8f}]".format(truth_loss))
            return truth_rank_list, gold_list, relevance_list
        else:
            rank_list, gold_list, relevance_list, loss = self.test_rnn(print_interval)
            print('Mean_Batch_loss [{0:<.8f}]'.format(loss))
            return rank_list, gold_list, relevance_list


