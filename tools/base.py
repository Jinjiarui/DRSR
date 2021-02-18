import os
import os.path as osp
import tensorflow as tf
import numpy as np
import shutil
import collections

from functools import reduce
from collections import namedtuple
from tools.click import *

class SummaryObj:
    """ Summary holder"""
    def __init__(self, log_dir, log_name, n_group=1, sess=None):
        self.name_set = set()
        self.n_group = n_group
        self.gra = None

        if sess is not None:
            self.sess = sess
            if os.path.exists(os.path.join(log_dir, log_name)):
                shutil.rmtree(os.path.join(log_dir, log_name))
            self.train_writer = tf.summary.FileWriter(log_dir + '/' + log_name, graph=tf.get_default_graph())

        else:
            self.gra = tf.Graph()
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess_config.gpu_options.allow_growth = True
            with self.gra.as_default():
                self.sess = tf.Session(graph=self.gra, config=sess_config)
                self.train_writer = tf.summary.FileWriter(log_dir + '/' + log_name, graph=tf.get_default_graph())
                self.sess.run(tf.global_variables_initializer())

    def register(self, name_list):
        """Register summary operations with a list contains names"""
        if self.gra is not None:
            with self.gra.as_default():
                for name in name_list:
                    if name in self.name_set:
                        raise Exception("You cannot define different operations with same name: `{}`".format(name))
                    self.name_set.add(name)
                    setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='agent_{}_{}'.format(i, name))
                                         for i in range(self.n_group)])
                    setattr(self, name + "_op", [tf.summary.scalar('agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                                 for i in range(self.n_group)])
        else:
            for name in name_list:
                if name in self.name_set:
                    raise Exception("You cannot define different operations with same name: `{}`".format(name))
                self.name_set.add(name)
                setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='agent_{}_{}'.format(i, name))
                                     for i in range(self.n_group)])
                setattr(self, name + "_op", [tf.summary.scalar('agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                             for i in range(self.n_group)])

    def write(self, summary_dict, step):
        """Write summary"""
        assert isinstance(summary_dict, dict)
        for key, value in summary_dict.items():
            if key not in self.name_set:
                raise Exception("Undefined operation: `{}`".format(key))
            if isinstance(value, list):
                for i in range(self.n_group):
                    self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[i],
                            feed_dict={getattr(self, key)[i]: value}), global_step=step)
                else:
                    self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[0],
                            feed_dict={getattr(self, key)[0]: value}), global_step=step)


class BaseModel(object):
    def __init__(self, sess, feature_space, name, batch_size):
        self.feature_space = feature_space
        self.name = name
        self.batch_size = batch_size
        self.sess = sess
        self.global_scope = None

    @property
    def gpu_config(self):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        return gpu_config

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.global_scope)

    def train(self, **kwargs):
        raise NotImplementedError

    def save(self, step, model_dir):
        model_dir = osp.join(model_dir, self.name)
        if not osp.exists(model_dir):
            os.makedirs(model_dir)
        assert self.sess is not None
        print('[INFO] Saving Model')
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.global_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, osp.join(model_dir, self.name), global_step=step)
        print('[INFO] Model Stored at: `{}`'.format(save_path))

    def load(self, step, model_dir):
        assert self.sess is not None
        save_path = osp.join(model_dir, self.name, self.name + '-' + str(step))
        print('[INFO] Restoring Model')
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.global_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, save_path)
        print('[INFO] Model Restored at `{}`'.format(save_path))


class Load_Data:
    def __init__(self, data_dir, data_pre, rank_cut=10, embed_size=700):
        self.initial_list = []
        self.initial_len = []
        self.query_id = []
        self.gold_list = []
        self.relevance = []
        self.feature = []
        self.document_id = []
        self.click = []
        self.observe_pro = []
        self.click_pro = []
        self.rank_list_size = rank_cut
        self.embed_size = embed_size
        # store data_dir and data_pre
        self.data_dir = data_dir
        self.data_pre = data_pre
        # split number in query-level
        self._query_id = []

    def load_feature_data(self):
        # <doc_id> <feature_id>:<feature_val> <feature_id>:<feature_val> ...
        # primitive feature for light_gbm
        primitive_feature = []
        feature_fin = open(self.data_dir + '/' + self.data_pre + '.feature')
        for line in feature_fin:
            _line = line.strip().split(' ')
            self.document_id.append(_line[0])
            primitive_feature.append(_line[1:])
            self.feature.append([0.0 for _ in range(self.embed_size)])
            for f in _line[1:]:
                _feature = f.split(":")
                # feature_id: _feature[0], feature_val: _feature[1]
                self.feature[-1][int(_feature[0])] = float(_feature[1])
        feature_fin.close()
        return self.feature, self.feature

    def load_document_data(self):
        # <click_label> <feature_id>:<feature_val> <feature_id>:<feature_val> ...
        # primitive feature for light_gbm
        primitive_feature = []
        feature_fin = open(self.data_dir + '/' + self.data_pre + '.feature')
        query_feature, query_primitive_feature, query_relevance = [], [], []
        _idx = 0
        for line in feature_fin:
            _line = line.strip().split(' ')
            if int(self._query_id[_idx]) == 0 and len(query_relevance) != 0:
                assert len(query_relevance) == len(query_feature)
                # assert len(query_relevance) == len(query_primitive_feature)
                self.relevance.append(query_relevance)
                self.feature.append(query_feature)
                # primitive_feature.append(query_primitive_feature)
                query_relevance = []
                query_feature = []
                # query_primitive_feature = []
            query_relevance.append(int(float(_line[0])))
            # query_primitive_feature.append(_line[1:])
            query_feature.append([0.0 for _ in range(self.embed_size)])
            for f in _line[1:]:
                _feature = f.split(":")
                query_feature[-1][int(_feature[0])] = float(_feature[1])
            _idx += 1
        feature_fin.close()
        return self.feature, self.feature, self.relevance

    def load_query_data(self):
        # <document_id in query>
        rank_fin = open(self.data_dir + '/' + self.data_pre + '.rank')
        query_initial = []
        for line in rank_fin:
            _line = line.strip().split(' ')
            if int(_line[0]) == 0 and len(query_initial) != 0:
                self.initial_list.append(query_initial)
                self.initial_len.append(len(query_initial))
                query_initial = []
            self._query_id.append(int(_line[0]))
            self.query_id.append(len(self._query_id) - 1)
            query_initial.append(int(_line[0]))
        rank_fin.close()
        return self.initial_len, self.query_id, self.initial_list

    def load_initial_data(self):
        initial_fin = open(self.data_dir + '/' + self.data_pre + '.init_list')
        # <query_id> <feature_line_number_for_the_1st_doc> <feature_line_number_for_the_2nd_doc> ...
        for line in initial_fin:
            _line = line.strip().split(' ')
            self.query_id.append(_line[0])
            self.initial_list.append([int(nb) for nb in _line[1:][:self.rank_list_size]])
        initial_fin.close()
        for _list in self.initial_list:
            self.initial_len.append(len(_list))
        return self.initial_len, self.query_id, self.initial_list

    def load_gold_data(self):
        # <query_id> <doc_idx_in_initial_list> <doc_idx_in_initial_list> ...
        gold_fin = open(self.data_dir + '/' + self.data_pre + '.gold_list')
        for line in gold_fin:
            _line = line.strip().split(' ')
            self.gold_list.append([int(idx) for idx in _line[1:][:self.rank_list_size]])
        gold_fin.close()
        return self.gold_list

    def load_relevance_data(self):
        # <query_id> <relevance_value_for_the_1st_doc> <relevance_value_for_the_2nd_doc> ...
        relevance_fin = open(self.data_dir + '/' + self.data_pre + '.weights')
        for line in relevance_fin:
            _line = line.strip().split(' ')
            self.relevance.append([float(val) for val in _line[1:][:self.rank_list_size]])
        relevance_fin.close()
        return self.relevance

    def generate_click_pro(self):
        click_model = PositionBiasedModel()
        for relevance in self.relevance:
            self.click_pro.append([click_model.get_click_probability(relevance=_r, relevance_level=4, position=_p)
                               for _r, _p in zip(relevance, range(len(relevance)))])
        # return click list
        return self.click_pro

    def generate_click_data(self, non_observe, click_observe):
        self.observe_pro, self.click_pro = self.generate_browse_data(non_observe, click_observe)
        for pro in self.click_pro:
            pro = np.array(pro)
            self.click.append((np.random.random_sample(pro.size) < pro).astype(np.int).tolist())
        # return click list
        return self.click

    def generate_browse_data(self, non_observe, click_observe):
        # non_observe for going observing after non-click
        # click_observe for going observing after click
        # set initial observe and click for item 0
        click_model = ClickClainModel(non_observe, click_observe)
        for relevance in self.relevance:
            iter_observe_list = []
            iter_click_list = []
            for iter_relevance in relevance:
                iter_observe = 1
                iter_click = 0
                iter_observe, iter_click = click_model.set_browse_probability(iter_observe, iter_click, iter_relevance, relevance_level=4)
                iter_observe_list.append(iter_observe)
                iter_click_list.append(iter_click)
            self.click_pro.append(iter_click_list)
            self.observe_pro.append(iter_observe_list)
        return self.observe_pro, self.click_pro

    def generate_rank_list(self, data, rerank_score):
        for qid in range(len(data.queryid)):
            assert len(data.initial_list[qid]) == len(data.gold_list[qid])
        if len(rerank_score) != len(data.initial_list):
            raise ValueError("Rerank ranklists number must be equal to initial list,"
                             " %d != %d." % (len(rerank_score), len(data.initial_list)))
        queryid_map = {}
        for qid in range(len(data.queryid)):
            _score = rerank_score[qid]
            rerank_list = sorted(range(len(_score)), key=lambda k: _score[k], reverse=True)
            if len(rerank_list) != len(data.gold_list[qid]):
                raise ValueError("Rerank ranklists length must be equal to gold list,"
                                 " %d != %d." % (len(rerank_list[qid]), len(data.gold_list[qid])))
            # remove duplicate and reorganize rerank list
            index_list = []
            index_set = set()
            for _idx in rerank_list[qid]:
                if _idx not in index_set:
                    index_set.add(_idx)
                    index_list.append(_idx)
            for _i in range(len(rerank_list)):
                if _i not in index_set:
                    index_list.append(_i)
            # get new ranking list
            query_id = data.queryid[qid]
            document_id = []
            for _i in index_list:
                _ni = data.initial_list[qid][_i]
                _ns = rerank_score[_i]
                if _ni >= 0:
                    document_id.append((data.documentid[_ni], _ns))
            queryid_map[query_id] = document_id
        return queryid_map

    def generate_final_list(self, data, rerank_score, output_dir):
        queryidmap = self.generate_rank_list(data, rerank_score)
        fout = open(output_dir + '.ranklist', 'w')
        for qid in data.queryid:
            for _idx in range(len(queryidmap[qid])):
                fout.write(qid + 'query_begin' + queryidmap[qid][_idx][0] + ' ' + str(_idx+1)
                           + " " + str(queryidmap[qid][_idx][1] + 'query_end'))
        fout.close()


def calculate_metrics(final_list, gold_list, relevance, scope_number):
    NDCG = []
    MAP = []
    for _final, _gold, _relevance in zip(final_list, gold_list, relevance):
        ideal_DCG = 0
        DCG = 0
        AP_value = 0
        AP_count = 0
        # define scope for calculation
        scope_final = _final[:scope_number]
        scope_gold = _gold[:scope_number]
        for _i, _f, _g in zip(range(1, scope_number + 1), scope_final, scope_gold):
            # calculate NDCG
            if _g == -1:
                break
            DCG += (pow(2, _relevance[_f]) - 1) / (np.log2(_i + 1))
            ideal_DCG += (pow(2, _relevance[_g]) - 1) / (np.log2(_i + 1))
            # calculate MAP
            if _relevance[_f] >= 1:
                AP_count += 1
                AP_value += AP_count / _i
        _NDCG = DCG / ideal_DCG if ideal_DCG != 0 else 0
        _MAP = AP_value / AP_count if AP_count != 0 else 0
        NDCG.append(_NDCG)
        MAP.append(_MAP)
    return NDCG, MAP


def calculate_metrics_gbm(click, relevance, document, gold, scope_number):
    NDCG, MAP = [], []
    # query-level click and relevance
    query_click, query_relevance, query_gold = [], [], []
    for _click, _relevance, _document, _gold in zip(click, relevance, document, gold):
        if _document == 0:
            if len(query_click) != 0 and len(query_relevance) != 0 and len(query_gold) != 0:
                assert len(query_click) == len(query_relevance)
                assert len(query_relevance) == len(query_gold)
                # calculate final list
                query_list = list(range(len(query_click)))
                query_map = zip(query_list, query_click)
                query_map = sorted(query_map, key=lambda d: d[1], reverse=False)
                query_list = list(zip(*query_map))[0]
                # initialize
                ideal_DCG = 0.
                DCG = 0.
                AP_value = 0.
                AP_count = 0.
                # define scope for calculation
                scope_list = query_list[:scope_number]
                scope_gold = query_gold[:scope_number]
                for _i, _l, _g in zip(range(1, scope_number + 1), scope_list, scope_gold):
                    # calculate NDCG
                    if _g == -1:
                        break
                    DCG += (pow(2, query_relevance[_l]) - 1) / (np.log2(_i + 1))
                    ideal_DCG += (pow(2, query_relevance[_g]) - 1) / (np.log2(_i + 1))
                    # calculate MAP
                    if query_relevance[_l] >= 1:
                        AP_count += 1
                        AP_value += AP_count / _i
                _NDCG = DCG / ideal_DCG if ideal_DCG != 0 else 0
                _MAP = AP_value / AP_count if AP_count != 0 else 0
                NDCG.append(_NDCG)
                MAP.append(_MAP)
                query_click, query_relevance, query_gold = [], [], []
        query_click.append(_click)
        query_relevance.append(_relevance)
        query_gold.append(_gold)
    return NDCG, MAP

