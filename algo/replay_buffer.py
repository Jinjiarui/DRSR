import numpy as np
import copy
import sys

sys.path.append('../')
from collections import namedtuple
from tools.buffer import LoopBuffer
Transition = namedtuple('Transition', 'feature, label, pair, censor_len, event_len, relevance, gold_list, init_list, click')

def extract_episode(dict_loop_buffer):
    data = []
    assert isinstance(dict_loop_buffer, dict)
    for transitions in dict_loop_buffer.values():
        episode = transitions.episode()
        if episode is not None:
            data.extend(transitions.episode())
    return data

class Episode(object):
    def __init__(self):
        self.feature = []
        self.label = []
        self.pair = []
        self.censor_len = []
        self.event_len = []
        self.relevance = []
        self.gold_list = []
        self.init_list = []
        self.click = []

    def append_feature(self, feature):
        self.feature = feature

    def append_label(self, label):
        self.label = label

    def append_pair(self, pair):
        self.pair = pair

    def append_censor(self, censor_len):
        self.censor_len = censor_len

    def append_event(self, event_len):
        self.event_len = event_len

    def append_relevance(self, relevance):
        self.relevance = relevance

    def append_gold(self, gold_list):
        self.gold_list = gold_list

    def append_init(self, init_list):
        self.init_list = init_list

    def append_click(self, click):
        self.click = click

    def episode(self):
        # assert len(self.state) == len(self.kl) ==len(self.entropy) \
        #        == len(self.reward) == len(self.action) == len(self.message) == len(self.goal)
        length = len(self.feature)
        data = []
        for i in range(length):
            meta = Transition(self.feature, self.label, self.pair,  self.censor_len,
                              self.event_len, self.relevance, self.gold_list, self.init_list, self.click)
            data.append(meta)
        return data


class WorkerBuffer(object):
    def __init__(self, max_len, use_priority=False):
        self._feature = LoopBuffer(max_len)
        self._label = LoopBuffer(max_len)
        self._pair = LoopBuffer(max_len)
        self._censor_len = LoopBuffer(max_len)
        self._event_len = LoopBuffer(max_len)
        self._relevance = LoopBuffer(max_len)
        self._gold_list = LoopBuffer(max_len)
        self._init_list = LoopBuffer(max_len)
        self._click = LoopBuffer(max_len)
        self._new_add = 0
        # index of array
        self._idx_arr = None

        self._tuple = namedtuple('Buffer', 'feature, label, pair, censor_len, event_len, relevance, gold_list, init_list, click')

    def __len__(self):
        return len(self._feature)

    @property
    def once_new_add(self):
        new_add = self._new_add
        self._new_add = 0
        return new_add

    def append(self, data):
        for transition in data:
            assert isinstance(transition, Transition)
            self._feature.append(transition.feature)
            self._label.append(transition.label)
            self._pair.append(transition.pair)
            self._censor_len.append(transition.censor_len)
            self._event_len.append(transition.event_len)
            self._relevance.append(transition.relevance)
            self._gold_list.append(transition.gold_list)
            self._init_list.append(transition.init_list)
            self._click.append(transition.click)
            self._new_add += 1

    def sample(self, batch_size):
        if self._idx_arr is None:
            self._idx_arr = np.array([_idx for _idx in range(batch_size)])
        # self._idx_arr = np.random.choice(len(self._feature), batch_size)
        self._idx_arr = self._idx_arr + batch_size if np.max(self._idx_arr) < len(self._feature) - batch_size else np.array([_idx for _idx in range(batch_size)])
        return self._tuple(
            feature=self._feature.sample(self._idx_arr),
            label=self._label.sample(self._idx_arr),
            pair=self._pair.sample(self._idx_arr),
            censor_len=self._censor_len.sample(self._idx_arr),
            event_len=self._event_len.sample(self._idx_arr),
            relevance=self._relevance.sample(self._idx_arr),
            gold_list=self._gold_list.sample(self._idx_arr),
            init_list=self._init_list.sample(self._idx_arr),
            click=self._click.sample(self._idx_arr)
        )

    def pop(self, batch_size):
        # pop the batch data using index of array
        self._feature.pop(self._idx_arr)
        self._label.pop(self._idx_arr)
        self._pair.pop(self._idx_arr)
        self._censor_len.pop(self._idx_arr)
        self._event_len.pop(self._idx_arr)
        self._relevance.pop(self._idx_arr)
        self._gold_list.pop(self._idx_arr)
        self._init_list.pop(self._idx_arr)
        self._click.pop(self._idx_arr)
        self._idx_arr -= batch_size

