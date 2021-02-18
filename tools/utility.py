import copy
from tools.base import *
from algo.replay_buffer import *
from tools.buffer import *
import pickle
import math
from tqdm import tqdm


def generate_query_data(data, embed_size, non_observe=0.00, click_observe=0.00, query_id_len=8, only_click=False):
    """generate query_data and feed into replay buffer, generate single click data ----------- Alibaba"""
    assert isinstance(data, Load_Data)
    # calculate query_num, event_len, censor_len, and relevance
    query_num, multi_num, average_event_len, average_censor_len, average_relevance = 0, 0, [], [], []
    data_len, data_query, data_initial = data.load_query_data()
    data_feature, data_primitive_feature, data_relevance = data.load_document_data()
    data_click = data.generate_click_data(non_observe=non_observe, click_observe=click_observe)
    # generate input for lightGBM
    data_click_gbm, data_feature_gbm, data_document_gbm, data_relevance_gbm, data_gold_gbm = [], [], [], [], []
    # initial replay_buffer
    replay_buffer = dict()
    for _len, _query, _init, _click, _relevance, _feature in zip(data_len, data_query, data_initial, data_click, data_relevance, data_feature):
        # generate single click data
        new_query_num = np.sum(_click)
        if only_click and new_query_num == 0:
            continue
        _idx = 1
        while _idx < new_query_num:
            multi_num += 1
            # record index of click
            _record_click_idx = []
            _click_index = list(range(len(_click)))
            for _i, _c in zip(_click_index, _click):
                if _c == 1:
                    _record_click_idx.append(_i)
            assert len(_record_click_idx) == new_query_num
            # format query
            _add_query = query_id_len - len(str(_query))
            _add_query_str = str()
            _query_idx = 0
            while _query_idx < _add_query:
                _add_query_str += 'X'
                _query_idx += 1
            _query_temp = str() + str(_query) + str(_idx)
            _click_idx_temp = _record_click_idx.copy()
            # generate list for rest click
            _click_idx_temp.remove(_record_click_idx[_idx-1])
            _relevance_temp = _relevance.copy()
            _click_temp = _click.copy()
            _init_temp = _init.copy()
            _feature_temp = _feature.copy()
            _click_idx_temp_list = list(range(len(_click_idx_temp)))
            for _idx_temp_list, _idx_temp in zip(_click_idx_temp_list, _click_idx_temp):
                _relevance_temp.pop(_idx_temp-_idx_temp_list)
                _click_temp.pop(_idx_temp-_idx_temp_list)
                _init_temp.pop(_idx_temp-_idx_temp_list)
                _feature_temp.pop(_idx_temp-_idx_temp_list)
            # feed into replay buffer
            if replay_buffer.get(_query_temp) is None:
                replay_buffer[_query_temp] = Episode()
            # generate gold list
            _gold_list_temp = list(range(len(_relevance_temp)))
            _gold_map_dict_temp = zip(_gold_list_temp, _relevance_temp)
            # sorted by value
            _gold_map_dict_temp = sorted(_gold_map_dict_temp, key=lambda d: d[1], reverse=True)
            _gold_list_temp = list(list(zip(*_gold_map_dict_temp))[0])
            _feature_temp_padding, _gold_list_temp_padding, _relevance_temp_padding, _click_temp_padding, _init_temp_padding = data_padding(rank_list_size=10, feature=_feature_temp, relevance=_relevance_temp, gold_list=_gold_list_temp, click=_click_temp, init_list=_init_temp, embed_size=embed_size)
            replay_buffer[_query_temp].append_feature(_feature_temp_padding)
            replay_buffer[_query_temp].append_gold(_gold_list_temp_padding)
            replay_buffer[_query_temp].append_relevance(_relevance_temp_padding)
            replay_buffer[_query_temp].append_click(_click_temp_padding)
            replay_buffer[_query_temp].append_init(_init_temp_padding)
            #  calculate label: [1, 0] for non-click [0, 1] for click
            assert np.sum(_click_temp) == 0 or np.sum(_click_temp) == 1
            if np.sum(_click_temp) == 0:
                _label_temp = [1, 0]
                _event_len_temp = -1
            else:
                _label_temp = [0, 1]
                _event_len_temp = 0
                assert np.sum(_click_temp) == 1
                for _c in _click_temp:
                    if _c == 1:
                        break
                    _event_len_temp += 1
            _len_temp = _len - new_query_num + 1
            replay_buffer[_query_temp].append_pair([0, 0])
            replay_buffer[_query_temp].append_label(_label_temp)
            replay_buffer[_query_temp].append_censor(_len_temp)
            replay_buffer[_query_temp].append_event(_event_len_temp)
            # calculate data information: query_num, event_len, censor_len, and relevance
            query_num += 1
            if _label_temp == [0, 1]:
                average_event_len.append(_event_len_temp)
            average_censor_len.append(_len_temp)
            average_relevance.extend(_relevance_temp)
            for _idx_temp_gbm, _feature_temp_gbm, _relevance_temp_gbm, _gold_list_temp_gbm in zip(range(len(_feature_temp_padding)), _feature_temp_padding, _relevance_temp_padding, _gold_list_temp_padding):
                data_feature_gbm.append(_feature_temp_gbm)
                data_document_gbm.append(_idx_temp_gbm)
                data_relevance_gbm.append(_relevance_temp_gbm)
                data_gold_gbm.append(_gold_list_temp_gbm)
                if _idx_temp_gbm == _event_len_temp:
                    data_click_gbm.append(1)
                else:
                    data_click_gbm.append(0)
            _idx += 1
        if new_query_num <= 1:
            # feed into replay buffer
            if replay_buffer.get(_query) is None:
                replay_buffer[_query] = Episode()
            # generate gold list
            _gold_list = list(range(len(_relevance)))
            _gold_map_dict = zip(_gold_list, _relevance)
            # sorted by value
            _gold_map_dict = sorted(_gold_map_dict, key=lambda d: d[1], reverse=True)
            _gold_list = list(list(zip(*_gold_map_dict))[0])
            _feature_padding, _gold_list_padding, _relevance_padding, _click_padding, _init_padding = data_padding(rank_list_size=10, feature=_feature, relevance=_relevance, gold_list=_gold_list, click=_click, init_list=_init, embed_size=embed_size)
            replay_buffer[_query].append_init(_init_padding)
            replay_buffer[_query].append_feature(_feature_padding)
            replay_buffer[_query].append_gold(_gold_list_padding)
            replay_buffer[_query].append_relevance(_relevance_padding)
            replay_buffer[_query].append_click(_click_padding)
            #  calculate label: [1, 0] for non-click [0, 1] for click
            if np.sum(_click) == 0:
                _label = [1, 0]
                _event_len = -1
            else:
                _label = [0, 1]
                _event_len = 0
                for _c in _click:
                    if _c == 1:
                        break
                    _event_len += 1
            replay_buffer[_query].append_pair([0, 0])
            replay_buffer[_query].append_label(_label)
            replay_buffer[_query].append_censor(_len)
            replay_buffer[_query].append_event(_event_len)
            # calculate data information: query_num, event_len, censor_len, and relevance
            query_num += 1
            if _label == [0, 1]:
                average_event_len.append(_event_len)
            average_censor_len.append(_len)
            average_relevance.extend(_relevance)
            # calculate data_feature, data_click for light_gbm
            # calculate data_query for light_gbm
            # calculate data_document for light_gbm
            # print('================ CHECK ONLY CHECK ==============')
            # print('batch_feature', np.array(_feature_padding).shape)
            # print(_feature_padding)
            # print('batch_gold', np.array(_gold_list_padding).shape)
            # print(_gold_list_padding)
            # print('batch_relevance', np.array(_relevance_padding).shape)
            # print(_relevance_padding)
            # print('batch_click', np.array(_click_padding).shape)
            # print(_click_padding)
            # print('batch_init', np.array(_init_padding).shape)
            # print(_init_padding)
            # print('batch_label', np.array(_label).shape)
            # print(_label)
            # print('batch_censor', np.array(_len).shape)
            # print(_len)
            # print('batch_event', np.array(_event_len).shape)
            # print(_event_len)
            # print("=================== END ONLY CHECK ================")
            for _idx_gbm, _feature_gbm, _relevance_gbm, _gold_list_gbm in zip(range(len(_feature_padding)), _feature_padding, _relevance_padding, _gold_list_padding):
                data_feature_gbm.append(_feature_gbm)
                data_document_gbm.append(_idx_gbm)
                data_relevance_gbm.append(_relevance_gbm)
                data_gold_gbm.append(_gold_list_gbm)
                if _idx_gbm == _event_len:
                    data_click_gbm.append(1)
                else:
                    data_click_gbm.append(0)
    print("================== BEGIN DATA ANALYSIS ==================")
    print('all_query_number', query_num)
    print('click_query_number', len(average_event_len))
    print('average_censor_len', np.average(average_censor_len))
    print('average_event_len', np.average(average_event_len))
    print('average_relevance', np.average(average_relevance))
    print("================== END DATA ANALYSIS ====================")
    return replay_buffer, data_click_gbm, data_feature_gbm, data_document_gbm, data_relevance_gbm, data_gold_gbm


def generate_batch_data(data, embed_size, non_observe=0.00, click_observe=0.00, only_click=False):
    """generate batch and feed into replay buffer, generate single click data ---------------- Yahoo"""
    assert isinstance(data, Load_Data)
    # calculate query_num, event_len, censor_len, and relevance
    query_num, multi_num, average_event_len, average_censor_len, average_relevance = 0, 0, [], [], []
    # primitive_feature for light_gbm
    data_feature, data_primitive_feature = data.load_feature_data()
    data_relevance = data.load_relevance_data()
    data_click = data.generate_click_data(non_observe=non_observe, click_observe=click_observe)
    data_len, data_query, data_initial = data.load_initial_data()
    # generate input for lightGBM
    data_click_gbm, data_feature_gbm, data_document_gbm, data_relevance_gbm, data_gold_gbm = [], [], [], [], []
    # initial replay_buffer
    replay_buffer = dict()
    for _len, _query, _init, _click, _relevance in zip(data_len, data_query, data_initial, data_click, data_relevance):
        # generate single click data
        new_query_num = np.sum(_click)
        if only_click and new_query_num == 0:
            continue
        _idx = 1
        while _idx < new_query_num:
            multi_num += 1
            # record index of click
            _record_click_idx = []
            _click_index = list(range(len(_click)))
            for _i, _c in zip(_click_index, _click):
                if _c == 1:
                    _record_click_idx.append(_i)
            assert len(_record_click_idx) == new_query_num
            _query_temp = str(_query) + str(_idx)
            _click_idx_temp = _record_click_idx.copy()
            # generate list for rest click
            _click_idx_temp.remove(_record_click_idx[_idx-1])
            _relevance_temp = _relevance.copy()
            _click_temp = _click.copy()
            _init_temp = _init.copy()
            _click_idx_temp_list = list(range(len(_click_idx_temp)))
            for _idx_temp_list, _idx_temp in zip(_click_idx_temp_list, _click_idx_temp):
                _relevance_temp.pop(_idx_temp-_idx_temp_list)
                _click_temp.pop(_idx_temp-_idx_temp_list)
                _init_temp.pop(_idx_temp-_idx_temp_list)
            # feed into replay buffer
            if replay_buffer.get(_query_temp) is None:
                replay_buffer[_query_temp] = Episode()
            # generate gold list
            _gold_list_temp = list(range(len(_relevance_temp)))
            _gold_map_dict_temp = zip(_gold_list_temp, _relevance_temp)
            # sorted by value
            _gold_map_dict_temp = sorted(_gold_map_dict_temp, key=lambda d: d[1], reverse=True)
            _gold_list_temp = list(list(zip(*_gold_map_dict_temp))[0])
            _feature_temp, _primitive_temp = [], []
            for _i in _init_temp:
                _f = data_feature[_i]
                # print(np.array(_f).shape)
                _feature_temp.append(_f)
                _primitive_f = data_primitive_feature[_i]
                _primitive_temp.append(_primitive_f)
            _feature_temp_padding, _gold_list_temp_padding, _relevance_temp_padding, _click_temp_padding, _init_temp_padding = data_padding(rank_list_size=10, feature=_feature_temp, relevance=_relevance_temp, gold_list=_gold_list_temp, click=_click_temp, init_list=_init_temp, embed_size=embed_size)
            replay_buffer[_query_temp].append_feature(_feature_temp_padding)
            replay_buffer[_query_temp].append_gold(_gold_list_temp_padding)
            replay_buffer[_query_temp].append_relevance(_relevance_temp_padding)
            replay_buffer[_query_temp].append_click(_click_temp_padding)
            replay_buffer[_query_temp].append_init(_init_temp_padding)
            #  calculate label: [1, 0] for non-click [0, 1] for click
            assert np.sum(_click_temp) == 0 or np.sum(_click_temp) == 1
            if np.sum(_click_temp) == 0:
                _label_temp = [1, 0]
                _event_len_temp = -1
            else:
                _label_temp = [0, 1]
                _event_len_temp = 0
                assert np.sum(_click_temp) == 1
                for _c in _click_temp:
                    if _c == 1:
                        break
                    _event_len_temp += 1
            _len_temp = _len - new_query_num + 1
            replay_buffer[_query_temp].append_pair([0, 0])
            replay_buffer[_query_temp].append_label(_label_temp)
            replay_buffer[_query_temp].append_censor(_len_temp)
            replay_buffer[_query_temp].append_event(_event_len_temp)
            # calculate data information: query_num, event_len, censor_len, and relevance
            query_num += 1
            if _label_temp == [0, 1]:
                average_event_len.append(_event_len_temp)
            average_censor_len.append(_len_temp)
            average_relevance.extend(_relevance_temp)
            for _idx_temp_gbm, _feature_temp_gbm, _relevance_temp_gbm, _gold_list_temp_gbm in zip(range(len(_feature_temp_padding)), _feature_temp_padding, _relevance_temp_padding, _gold_list_temp_padding):
                data_feature_gbm.append(_feature_temp_gbm)
                data_document_gbm.append(_idx_temp_gbm)
                data_relevance_gbm.append(_relevance_temp_gbm)
                data_gold_gbm.append(_gold_list_temp_gbm)
                if _idx_temp_gbm == _event_len_temp:
                    data_click_gbm.append(1)
                else:
                    data_click_gbm.append(0)
            _idx += 1
        if new_query_num <= 1:
            # feed into replay buffer
            if replay_buffer.get(_query) is None:
                replay_buffer[_query] = Episode()
            # generate gold list
            _gold_list = list(range(len(_relevance)))
            _gold_map_dict = zip(_gold_list, _relevance)
            # sorted by value
            _gold_map_dict = sorted(_gold_map_dict, key=lambda d: d[1], reverse=True)
            _gold_list = list(list(zip(*_gold_map_dict))[0])
            _feature = []
            for _i in _init:
                _f = data_feature[_i]
                _feature.append(_f)
            _feature_padding, _gold_list_padding, _relevance_padding, _click_padding, _init_padding = data_padding(rank_list_size=10, feature=_feature, relevance=_relevance, gold_list=_gold_list, click=_click, init_list=_init, embed_size=embed_size)
            replay_buffer[_query].append_init(_init_padding)
            replay_buffer[_query].append_feature(_feature_padding)
            replay_buffer[_query].append_gold(_gold_list_padding)
            replay_buffer[_query].append_relevance(_relevance_padding)
            replay_buffer[_query].append_click(_click_padding)
            #  calculate label: [1, 0] for non-click [0, 1] for click
            if np.sum(_click) == 0:
                _label = [1, 0]
                _event_len = -1
            else:
                _label = [0, 1]
                _event_len = 0
                for _c in _click:
                    if _c == 1:
                        break
                    _event_len += 1
            replay_buffer[_query].append_pair([0, 0])
            replay_buffer[_query].append_label(_label)
            replay_buffer[_query].append_censor(_len)
            replay_buffer[_query].append_event(_event_len)
            # calculate data information: query_num, event_len, censor_len, and relevance
            query_num += 1
            if _label == [0, 1]:
                average_event_len.append(_event_len)
            average_censor_len.append(_len)
            average_relevance.extend(_relevance)
            # calculate data_feature, data_click for light_gbm
            # calculate data_query for light_gbm
            # calculate data_document for light_gbm
            # print('================ CHECK ONLY CHECK ==============')
            # print('batch_feature', np.array(_feature_padding).shape)
            # print(_feature_padding)
            # print('batch_gold', np.array(_gold_list_padding).shape)
            # print(_gold_list_padding)
            # print('batch_relevance', np.array(_relevance_padding).shape)
            # print(_relevance_padding)
            # print('batch_click', np.array(_click_padding).shape)
            # print(_click_padding)
            # print('batch_init', np.array(_init_padding).shape)
            # print(_init_padding)
            # print('batch_label', np.array(_label).shape)
            # print(_label)
            # print('batch_censor', np.array(_len).shape)
            # print(_len)
            # print('batch_event', np.array(_event_len).shape)
            # print(_event_len)
            # print("=================== END ONLY CHECK ================")
            for _idx_gbm, _feature_gbm, _relevance_gbm, _gold_list_gbm in zip(range(len(_feature_padding)), _feature_padding, _relevance_padding, _gold_list_padding):
                data_feature_gbm.append(_feature_gbm)
                data_document_gbm.append(_idx_gbm)
                data_relevance_gbm.append(_relevance_gbm)
                data_gold_gbm.append(_gold_list_gbm)
                if _idx_gbm == _event_len:
                    data_click_gbm.append(1)
                else:
                    data_click_gbm.append(0)
    print("================== BEGIN DATA ANALYSIS ==================")
    print('all_query_number', query_num)
    print('click_query_number', len(average_event_len))
    print('average_censor_len', np.average(average_censor_len))
    print('average_event_len', np.average(average_event_len))
    print('average_relevance', np.average(average_relevance))
    print("================== END DATA ANALYSIS ====================")
    return replay_buffer, data_click_gbm, data_feature_gbm, data_document_gbm, data_relevance_gbm, data_gold_gbm


#  permutation order technique for pair-wise setting
def generate_permutation_order(replay_buffer):
    isinstance(replay_buffer, dict)
    permutation_replay_buffer = dict()
    for _query, _episode in replay_buffer.items():
        isinstance(_episode, Episode)
        # select click data: [1, 0] for non-click [0, 1] for click
        if np.argmax(_episode.label):
            # generate data for re-rank zero
            # exchange the position between event and before
            for _before in range(1, _episode.event_len - 1):
                _query_temp = str(_query) + str(_before)
                if permutation_replay_buffer.get(_query_temp) is None:
                    permutation_replay_buffer[_query_temp] = Episode()
                _feature_temp = _episode.feature.copy()
                _relevance_temp = _episode.relevance.copy()
                _gold_temp = _episode.gold_list.copy()
                _init_temp = _episode.init_list.copy()
                _click_temp = _episode.click.copy()
                _feature_temp[_before], _feature_temp[_episode.event_len] = _feature_temp[_episode.event_len], _feature_temp[_before]
                _relevance_temp[_before], _relevance_temp[_episode.event_len] = _relevance_temp[_episode.event_len], _relevance_temp[_before]
                _gold_temp[_before], _gold_temp[_episode.event_len] = _gold_temp[_episode.event_len], _gold_temp[_before]
                _init_temp[_before], _init_temp[_episode.event_len] = _init_temp[_episode.event_len], _init_temp[_before]
                _click_temp[_before], _click_temp[_episode.event_len] = _click_temp[_episode.event_len], _click_temp[_before]
                permutation_replay_buffer[_query_temp].append_censor(_episode.censor_len)
                permutation_replay_buffer[_query_temp].append_event(_before)
                permutation_replay_buffer[_query_temp].append_gold(_gold_temp)
                permutation_replay_buffer[_query_temp].append_feature(_feature_temp)
                permutation_replay_buffer[_query_temp].append_relevance(_relevance_temp)
                permutation_replay_buffer[_query_temp].append_init(_init_temp)
                permutation_replay_buffer[_query_temp].append_click(_click_temp)
                # [0, 1] for before
                permutation_replay_buffer[_query_temp].append_pair([0, 1])
                permutation_replay_buffer[_query_temp].append_label(_episode.label.copy())
            # generate data for re-rank one
            # exchange the position between event and after
            for _after in range(_episode.event_len + 1, _episode.censor_len):
                _query_temp = str(_query) + str(_after)
                if permutation_replay_buffer.get(_query_temp) is None:
                    permutation_replay_buffer[_query_temp] = Episode()
                _feature_temp = _episode.feature.copy()
                _relevance_temp = _episode.relevance.copy()
                _gold_temp = _episode.gold_list.copy()
                _init_temp = _episode.init_list.copy()
                _click_temp = _episode.click.copy()
                _feature_temp[_after], _feature_temp[_episode.event_len] = _feature_temp[_episode.event_len], _feature_temp[_after]
                _relevance_temp[_after], _relevance_temp[_episode.event_len] = _relevance_temp[_episode.event_len], _relevance_temp[_after]
                _gold_temp[_after], _gold_temp[_episode.event_len] = _gold_temp[_episode.event_len], _gold_temp[_after]
                _init_temp[_after], _click_temp[_episode.event_len] = _click_temp[_episode.event_len], _click_temp[_after]
                permutation_replay_buffer[_query_temp].append_censor(_episode.censor_len)
                permutation_replay_buffer[_query_temp].append_event(_after)
                permutation_replay_buffer[_query_temp].append_gold(_gold_temp)
                permutation_replay_buffer[_query_temp].append_feature(_feature_temp)
                permutation_replay_buffer[_query_temp].append_relevance(_relevance_temp)
                permutation_replay_buffer[_query_temp].append_click(_click_temp)
                permutation_replay_buffer[_query_temp].append_init(_init_temp)
                permutation_replay_buffer[_query_temp].append_pair([1, 0])
                permutation_replay_buffer[_query_temp].append_label(_episode.label.copy())
            return permutation_replay_buffer


def data_padding(rank_list_size, feature, gold_list, relevance, click, init_list, embed_size):
    # padding zeros in tails of rnn model
    assert len(feature) == len(gold_list)
    assert len(feature) == len(relevance)
    assert len(feature) == len(init_list)
    new_padding_size = rank_list_size - len(feature)
    for _ in range(new_padding_size):
        feature.append([0 for _ in range(embed_size)])
        gold_list += [-1]
        relevance += [0]
        init_list += [-1]
    click_padding_size = rank_list_size - len(click)
    for _ in range(click_padding_size):
        click += [0]
    return feature, gold_list, relevance, click, init_list


def store_replay_buffer(data_dir, train_dir, test_dir, train_data_pre, test_data_pre, data_name, non_observe_pro=0.00, click_observe_pro=0.00, rank_len=10, embed_size=700, only_click=False):
    assert data_name == 'Yahoo' or data_name == 'Alibaba'
    print("============================= YOU ARE IN " + data_name + " =============================")
    data_train = Load_Data(data_dir=train_dir, data_pre=train_data_pre, rank_cut=rank_len, embed_size=embed_size)
    data_test = Load_Data(data_dir=test_dir, data_pre=test_data_pre, rank_cut=rank_len, embed_size=embed_size)
    if data_name == 'Yahoo':
        train_batch_replay_buffer, train_click_gbm, train_feature_gbm, train_document_gbm, train_relevance_gbm, train_gold_gbm = generate_batch_data(data_train, non_observe=non_observe_pro, click_observe=click_observe_pro, embed_size=embed_size, only_click=only_click)
        test_batch_replay_buffer, test_click_gbm, test_feature_gbm, test_document_gbm, test_relevance_gbm, test_gold_gbm = generate_batch_data(data_test, non_observe=non_observe_pro, click_observe=click_observe_pro, embed_size=embed_size, only_click=only_click)
    else:
        train_batch_replay_buffer, train_click_gbm, train_feature_gbm, train_document_gbm, train_relevance_gbm, train_gold_gbm = generate_query_data(data_train, non_observe=non_observe_pro, click_observe=click_observe_pro, embed_size=embed_size, only_click=only_click)
        test_batch_replay_buffer, test_click_gbm, test_feature_gbm, test_document_gbm, test_relevance_gbm, test_gold_gbm = generate_query_data(data_test, non_observe=non_observe_pro, click_observe=click_observe_pro, embed_size=embed_size, only_click=only_click)
    data_train = open(data_dir + '/' + train_data_pre + 'data_train', 'wb')
    pickle.dump(train_batch_replay_buffer, data_train)
    data_train.close()
    data_test = open(data_dir + '/' + test_data_pre + 'data_test', 'wb')
    pickle.dump(test_batch_replay_buffer, data_test)
    data_test.close()
    train_batch_pair_replay_buffer = generate_permutation_order(train_batch_replay_buffer)
    test_batch_pair_replay_buffer = generate_permutation_order(test_batch_replay_buffer)
    data_train_pair = open(data_dir + '/' + train_data_pre + 'data_train_pair', 'wb')
    pickle.dump(train_batch_pair_replay_buffer, data_train_pair)
    data_train_pair.close()
    data_test_pair = open(data_dir + '/' + test_data_pre + 'data_test_pair', 'wb')
    pickle.dump(test_batch_pair_replay_buffer, data_test_pair)
    data_test_pair.close()

    # generate LightGBM input: <label: bool for click> <feature>
    data_train_feature_gbm = open(data_dir + '/' + train_data_pre + 'data_train_feature_gbm', 'wb')
    pickle.dump(train_click_gbm, data_train_feature_gbm)
    pickle.dump(train_feature_gbm, data_train_feature_gbm)
    # for _train_click, _train_feature in zip(train_click_gbm, train_feature_gbm):
    #     data_train_feature_gbm.write(str(_train_click) + ' ' + str(_train_feature) + '\n')
    data_train_feature_gbm.close()
    data_train_document_gbm = open(data_dir + '/' + train_data_pre + 'data_train_document_gbm', 'wb')
    pickle.dump(train_document_gbm, data_train_document_gbm)
    # for _train_document in train_document_gbm:
    #     data_train_document_gbm.write(str(_train_document) + '\n')
    data_train_document_gbm.close()
    # data_train_query_gbm = open(data_dir + '/' + 'data_train_query_gbm', 'w')
    # for _train_query in train_query_gbm:
    #     data_train_query_gbm.write(str(_train_query) + '\n')
    # data_train_query_gbm.close()
    data_train_relevance_gbm = open(data_dir + '/' + train_data_pre + 'data_train_relevance_gbm', 'wb')
    pickle.dump(train_relevance_gbm, data_train_relevance_gbm)
    data_train_relevance_gbm.close()
    data_train_gold_gbm = open(data_dir + '/' + train_data_pre + 'data_train_gold_gbm', 'wb')
    pickle.dump(train_gold_gbm, data_train_gold_gbm)
    data_train_gold_gbm.close()

    data_test_feature_gbm = open(data_dir + '/' + test_data_pre + 'data_test_feature_gbm', 'wb')
    # for _test_click, _test_feature in zip(test_click_gbm, test_feature_gbm):
    #     data_test_feature_gbm.write(str(_test_click) + ' ' + str(_test_feature) + '\n')
    pickle.dump(test_click_gbm, data_test_feature_gbm)
    pickle.dump(test_feature_gbm, data_test_feature_gbm)
    data_test_feature_gbm.close()
    data_test_document_gbm = open(data_dir + '/' + test_data_pre + 'data_test_document_gbm', 'wb')
    # for _test_document in test_document_gbm:
    #     data_test_document_gbm.write(str(_test_document) + '\n')
    pickle.dump(test_document_gbm, data_test_document_gbm)
    data_test_document_gbm.close()
    # data_test_query_gbm = open(data_dir + '/' + 'data_test_query_gbm', 'w')
    # for _test_query in test_query_gbm:
    #     data_test_query_gbm.write(str(_test_query) + '\n')
    # data_test_query_gbm.close()
    data_test_relevance_gbm = open(data_dir + '/' + test_data_pre + 'data_test_relevance_gbm', 'wb')
    pickle.dump(test_relevance_gbm, data_test_relevance_gbm)
    data_test_relevance_gbm.close()
    data_test_gold_gbm = open(data_dir + '/' + test_data_pre + 'data_test_gold_gbm', 'wb')
    pickle.dump(test_gold_gbm, data_test_gold_gbm)
    data_test_gold_gbm.close()


def load_replay_buffer(data_dir, train_data_pre):
    assert isinstance(train_data_pre, str)
    test_data_pre = train_data_pre.replace('train', 'test')
    data_train = open(data_dir + '/' + train_data_pre + 'data_train', 'rb')
    train_batch_replay_buffer = pickle.load(data_train)
    data_train.close()
    data_test = open(data_dir + '/' + test_data_pre + 'data_test', 'rb')
    test_batch_replay_buffer = pickle.load(data_test)
    data_train.close()
    data_train_pair = open(data_dir + '/' + train_data_pre + 'data_train_pair', 'rb')
    train_batch_pair_replay_buffer = pickle.load(data_train_pair) if data_train_pair is not None else None
    data_train_pair.close()
    data_test_pair = open(data_dir + '/' + test_data_pre + 'data_test_pair', 'rb')
    test_batch_pair_replay_buffer = pickle.load(data_test_pair) if data_test_pair is not None else None
    data_test_pair.close()
    return train_batch_replay_buffer, test_batch_replay_buffer, train_batch_pair_replay_buffer, test_batch_pair_replay_buffer


# add element in stored replay buffer for EM
def generate_batch_replay_buffer(data, embed_size, data_click):
    assert isinstance(data, Load_Data)
    # primitive_feature for light_gbm
    data_feature, data_primitive_feature = data.load_feature_data()
    data_relevance = data.load_relevance_data()
    data_len, data_query, data_initial = data.load_initial_data()
    # generate input for lightGBM
    data_click_gbm, data_feature_gbm, data_document_gbm, data_query_gbm = [], [], [], []
    # initial replay_buffer
    replay_buffer = dict()
    for _len, _query, _init, _click, _relevance in zip(data_len, data_query, data_initial, data_click, data_relevance):
        if replay_buffer.get(_query) is None:
            replay_buffer[_query] = Episode()
        # generate gold list
        _gold_list = list(range(len(_relevance)))
        _gold_map_dict = zip(_gold_list, _relevance)
        # sorted by value
        _gold_map_dict = sorted(_gold_map_dict, key=lambda d: d[1], reverse=True)
        _gold_list = list(list(zip(*_gold_map_dict))[0])
        _feature = []
        for _i in _init:
            _f = data_feature[_i]
            _feature.append(_f)
        _feature_padding, _gold_list_padding, _relevance_padding, _click_padding, _init_list_padding = data_padding(rank_list_size=10, feature=_feature, relevance=_relevance, gold_list=_gold_list, click=_click, init_list=_init, embed_size=embed_size)
        replay_buffer[_query].append_feature(_feature_padding)
        replay_buffer[_query].append_gold(_gold_list_padding)
        replay_buffer[_query].append_relevance(_relevance_padding)
        replay_buffer[_query].append_click(_click_padding)
        replay_buffer[_query].append_init(_init_list_padding)
        #  calculate label: [1, 0] for non-click [0, 1] for click
        if np.sum(_click) == 0:
            _label = [1, 0]
            _event_len = -1
        else:
            _label = [0, 1]
            _event_len = 0
            for _c in _click:
                if _c == 1:
                    break
                _event_len += 1
        replay_buffer[_query].append_pair([0, 0])
        replay_buffer[_query].append_label(_label)
        replay_buffer[_query].append_censor(_len)
        replay_buffer[_query].append_event(_event_len)
        # calculate data_feature, data_click for light_gbm
        # calculate data_query for light_gbm
        # calculate data_document for light_gbm
        for _idx_gbm, _feature_gbm in zip(range(len(_feature)), _feature):
            data_feature_gbm.append(_feature_gbm)
            data_document_gbm.append(_idx_gbm)
            if _idx_gbm == _event_len:
                data_click_gbm.append(1)
            else:
                data_click_gbm.append(0)
        data_query_gbm.append(len(_feature))
    return replay_buffer, data_click_gbm, data_feature_gbm, data_document_gbm, data_query_gbm
