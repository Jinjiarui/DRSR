import tensorflow as tf
import numpy as np
import random
import sys
import lightgbm as lgb

sys.path.append('../')

from algo.replay_buffer import WorkerBuffer
from tools.base import BaseModel


class TREE(BaseModel):
    def __init__(self, sess=None, feature_space=None, name='TREE', batch_size=None, learning_rate=0.1,
                 num_leaves=32):
        super(TREE, self).__init__(sess, feature_space, name, batch_size)

        self._lr = learning_rate
        self.params = {'num_leaves': num_leaves, 'learning_rate': learning_rate, 'use_two_round_loading': 'false',
                       'objective': 'multiclass', 'boost_from_average': 'true', 'num_class': 5}
        self.data = None
        self.model = None

    def build(self, data_feature, data_label):
        self.data = lgb.Dataset(data_feature, data_label, free_raw_data=False)
        self.data.construct()
        return self.data

    def update_label(self, label):
        """Update new label"""
        assert isinstance(self.data, lgb.Dataset)
        feature = self.data.get_data()
        add_data = lgb.Dataset(feature, label, free_raw_data=False)
        add_data.construct()
        self.data.add_features_from(add_data)

    def train(self, data_feature):
        self.model = lgb.train(self.params, self.data, init_model=self.model, num_boost_round=50)
        return np.argmax(self.model.predict(data_feature), axis=1)

    def test(self, data_feature, data_label):
        _data = lgb.Dataset(data_feature, data_label)
        _data.construct()
        # return self.model.eval(_data)
        return np.argmax(self.model.predict(data_feature), axis=1)


