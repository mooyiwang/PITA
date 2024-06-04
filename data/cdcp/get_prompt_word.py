from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json
import pandas as pd
from models.pos_map_cdcp import pair2sequence, pair_idx_map
import random


# model_name_or_path = "roberta-large"
#
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# def split_label_words(tokenizer, label_list):
#     label_word_list = []
#     for label in label_list:
#         if label == 'no_relation' or label == "NA":
#             label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
#             label_word_list.append(torch.tensor(label_word_id))
#         else:
#             tmps = label
#             label = label.lower()
#             label = label.split("(")[0]
#             label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
#             label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
#             print(label, label_word_id)
#             label_word_list.append(torch.tensor(label_word_id))
#     padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
#     return padded_label_word_list


data_df = pd.read_csv("cdcp_data_df2.csv")

split_test_file_path = "./data/cdcp/cdcp_test_index.json"
with open(split_test_file_path, "r") as fp:
    test_id_list = json.load(fp)
test_data_df = data_df[data_df["para_id"].isin(test_id_list)]
train_data_df = data_df[~(data_df["para_id"].isin(test_id_list))]

essay_id2parag_id_dict = train_data_df.groupby("essay_id").groups
essay_id_list = list(essay_id2parag_id_dict.keys())
random.seed(42)
random.shuffle(essay_id_list)
num_train_essay = int(len(essay_id_list) * 0.9)
dev_essay_id_list = essay_id_list[num_train_essay:]
dev_para_id_list = []
for essay_id in dev_essay_id_list:
    dev_para_id_list += essay_id2parag_id_dict[essay_id].tolist()

dev_data_df = train_data_df[train_data_df["para_id"].isin(dev_para_id_list)]
train_data_df = train_data_df[~train_data_df["para_id"].isin(dev_para_id_list)]

train_data_df = train_data_df[train_data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]
dev_data_df = dev_data_df[dev_data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]
test_data_df = test_data_df[test_data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]


def get_label_dsitribution(data_df):
    AC_types_list = [list(eval(AC_types)) for AC_types in data_df["ac_types"]]
    AR_pairs_list = [eval(_) for _ in data_df['ac_rel_pairs']] # ac_types,ac_rel_targets,ac_rel_types,ac_rel_pairs
    AR_types_list = [eval(_) for _ in data_df['ac_rel_types']]

    ac_type_map = {"value": 0, "policy": 0, "testimony": 0, "fact": 0, "reference": 0}
    for AC_types in AC_types_list:
        if len(AC_types) == 0:
            continue
        for type in AC_types:
            ac_type_map[type] += 1

    ar_map = {'no relation':0 , 'relation': 0}
    for i, AR_pairs in enumerate(AR_pairs_list):
        ac_num = len(AC_types_list[i])
        if ac_num == 1 or ac_num == 0:
            continue
        rel_num = len(AR_pairs)
        # print("ac_num", ac_num)
        no_rel_num = len(pair_idx_map[ac_num]) - rel_num
        ar_map['no relation'] += no_rel_num
        ar_map['relation'] += rel_num

    ar_type_map = {"reason": 0, "evidence": 0}
    for i, AR_types in enumerate(AR_types_list):
        ar_num = len(AR_types)
        if ar_num == 0:
            continue
        for type in AR_types:
            ar_type_map[type] += 1

    print("ac_type_map", ac_type_map)
    print("ar_map", ar_map)
    print("ar_type_map", ar_type_map)

# data_df:
# ac_type_map {'value': 2158, 'policy': 810, 'testimony': 1026, 'fact': 746, 'reference': 32}
# ar_map {'no relation': 18455, 'relation': 1352}
# ar_type_map {'reason': 1306, 'evidence': 46}

get_label_dsitribution(train_data_df)
get_label_dsitribution(dev_data_df)
get_label_dsitribution(test_data_df)

# train_df:
# ac_type_map {'value': 1729, 'policy': 673, 'testimony': 771, 'fact': 589, 'reference': 27}
# ar_map {'no relation': 13831, 'relation': 1108}
# ar_type_map {'reason': 1067, 'evidence': 41}
# dev_df:
# ac_type_map {'value': 214, 'policy': 68, 'testimony': 115, 'fact': 82, 'reference': 4}
# ar_map {'no relation': 2260, 'relation': 124}
# ar_type_map {'reason': 122, 'evidence': 2}
# test_df:
# ac_type_map {'value': 215, 'policy': 69, 'testimony': 140, 'fact': 75, 'reference': 1}
# ar_map {'no relation': 2364, 'relation': 120}
# ar_type_map {'reason': 117, 'evidence': 3}

# ac_type_idx_map = { i : {"value": 0, "policy": 0, "testimony": 0, "fact": 0, "reference": 0} for i in range(28)}
# ar_idx_map = {pair: {'no relation':0 , 'relation': 0} for pair in list(pair2sequence[28])}
# ar_type_idx_map = {pair: {"reason": 0, "evidence": 0} for pair in list(pair2sequence[28])}
# print(len(AC_types_list), len(AR_pairs_list), len(AR_types_list))
# for AC_types, AR_pairs, AR_types in zip(AC_types_list, AR_pairs_list, AR_types_list):
#     ac_num = len(AC_types)
#     if ac_num == 0:
#         continue
#
#     for j, type in enumerate(AC_types):
#         ac_type_idx_map[j][type] += 1
#
#     if ac_num == 1:
#         continue
#     for pair, type in zip(AR_pairs, AR_types):
#         p1, p2 = min(pair), max(pair)
#         if abs(p1 - p2) > 9:
#             continue
#         ar_idx_map[(p1, p2)]['relation'] += 1
#         ar_type_idx_map[(p1, p2)][type] += 1
#
#     for pair in list(pair2sequence[ac_num]):
#         if pair not in AR_pairs and (pair[1], pair[0]) not in AR_pairs:
#             ar_idx_map[pair]['no relation'] += 1
#
# print("ac_type_idx_map", ac_type_idx_map)
# print("ar_idx_map", ar_idx_map)
# print("ar_type_idx_map", ar_type_idx_map)


# ac_type_idx_map = {0: {'value': 328, 'policy': 147, 'testimony': 150, 'fact': 105, 'reference': 1},
# 1: {'value': 299, 'policy': 121, 'testimony': 163, 'fact': 146, 'reference': 2},
# 2: {'value': 254, 'policy': 116, 'testimony': 133, 'fact': 125, 'reference': 2},
# 3: {'value': 246, 'policy': 75, 'testimony': 97, 'fact': 90, 'reference': 4},
# 4: {'value': 186, 'policy': 67, 'testimony': 83, 'fact': 68, 'reference': 3},
# 5: {'value': 151, 'policy': 67, 'testimony': 65, 'fact': 45, 'reference': 3},
# 6: {'value': 131, 'policy': 49, 'testimony': 47, 'fact': 40, 'reference': 2},
# 7: {'value': 109, 'policy': 30, 'testimony': 48, 'fact': 26, 'reference': 2},
# 8: {'value': 84, 'policy': 32, 'testimony': 41, 'fact': 21, 'reference': 0},
# 9: {'value': 73, 'policy': 22, 'testimony': 30, 'fact': 19, 'reference': 1},
# 10: {'value': 62, 'policy': 13, 'testimony': 25, 'fact': 15, 'reference': 1},
# 11: {'value': 44, 'policy': 14, 'testimony': 29, 'fact': 8, 'reference': 0},
# 12: {'value': 44, 'policy': 7, 'testimony': 17, 'fact': 6, 'reference': 1},
# 13: {'value': 32, 'policy': 10, 'testimony': 17, 'fact': 6, 'reference': 1},
# 14: {'value': 19, 'policy': 9, 'testimony': 13, 'fact': 5, 'reference': 2},
# 15: {'value': 19, 'policy': 4, 'testimony': 8, 'fact': 5, 'reference': 1},
# 16: {'value': 14, 'policy': 3, 'testimony': 12, 'fact': 3, 'reference': 0},
# 17: {'value': 10, 'policy': 5, 'testimony': 9, 'fact': 2, 'reference': 1},
# 18: {'value': 9, 'policy': 3, 'testimony': 8, 'fact': 2, 'reference': 1},
# 19: {'value': 7, 'policy': 2, 'testimony': 8, 'fact': 3, 'reference': 1},
# 20: {'value': 9, 'policy': 1, 'testimony': 5, 'fact': 2, 'reference': 1},
# 21: {'value': 5, 'policy': 4, 'testimony': 5, 'fact': 2, 'reference': 1},
# 22: {'value': 7, 'policy': 3, 'testimony': 4, 'fact': 1, 'reference': 0},
# 23: {'value': 3, 'policy': 2, 'testimony': 5, 'fact': 0, 'reference': 1},
# 24: {'value': 4, 'policy': 2, 'testimony': 2, 'fact': 0, 'reference': 0},
# 25: {'value': 3, 'policy': 0, 'testimony': 2, 'fact': 1, 'reference': 0},
# 26: {'value': 4, 'policy': 1, 'testimony': 0, 'fact': 0, 'reference': 0},
# 27: {'value': 2, 'policy': 1, 'testimony': 0, 'fact': 0, 'reference': 0}}
#
# ac_type_idx_p_map = {}
# for key, value in ac_type_idx_map.items():
#     n1, n2, n3, n4, n5 = value["value"], value["policy"], value["testimony"], value["fact"], value["reference"]
#     n = n1 + n2 + n3 + n4 + n5
#     ac_type_idx_p_map[key] = tuple([n1 / n, n2 / n, n3 / n, n4 / n, n5 / n])
# print("ac_type_idx_p_map", ac_type_idx_p_map)
#
#
# ar_idx_map = {(0, 1): {'no relation': 500, 'relation': 231}, (0, 2): {'no relation': 537, 'relation': 93},
#               (0, 3): {'no relation': 472, 'relation': 40}, (0, 4): {'no relation': 388, 'relation': 19},
#               (0, 5): {'no relation': 324, 'relation': 7}, (0, 6): {'no relation': 267, 'relation': 2},
#               (0, 7): {'no relation': 211, 'relation': 4}, (0, 8): {'no relation': 174, 'relation': 4},
#               (0, 9): {'no relation': 143, 'relation': 2}, (0, 10): {'no relation': 116, 'relation': 0},
#               (0, 11): {'no relation': 95, 'relation': 0}, (0, 12): {'no relation': 74, 'relation': 0},
#               (1, 2): {'no relation': 507, 'relation': 123}, (1, 3): {'no relation': 483, 'relation': 29},
#               (1, 4): {'no relation': 392, 'relation': 15}, (1, 5): {'no relation': 327, 'relation': 4},
#               (1, 6): {'no relation': 267, 'relation': 2}, (1, 7): {'no relation': 213, 'relation': 2},
#               (1, 8): {'no relation': 177, 'relation': 1}, (1, 9): {'no relation': 144, 'relation': 1},
#               (1, 10): {'no relation': 116, 'relation': 0}, (1, 11): {'no relation': 95, 'relation': 0},
#               (1, 12): {'no relation': 74, 'relation': 0}, (1, 13): {'no relation': 66, 'relation': 0},
#               (2, 3): {'no relation': 384, 'relation': 128}, (2, 4): {'no relation': 372, 'relation': 35},
#               (2, 5): {'no relation': 322, 'relation': 9}, (2, 6): {'no relation': 264, 'relation': 5},
#               (2, 7): {'no relation': 213, 'relation': 2}, (2, 8): {'no relation': 178, 'relation': 0},
#               (2, 9): {'no relation': 145, 'relation': 0}, (2, 10): {'no relation': 116, 'relation': 0},
#               (2, 11): {'no relation': 95, 'relation': 0}, (2, 12): {'no relation': 74, 'relation': 0},
#               (2, 13): {'no relation': 66, 'relation': 0}, (2, 14): {'no relation': 47, 'relation': 0},
#               (3, 4): {'no relation': 309, 'relation': 98}, (3, 5): {'no relation': 308, 'relation': 23},
#               (3, 6): {'no relation': 261, 'relation': 8}, (3, 7): {'no relation': 212, 'relation': 3},
#               (3, 8): {'no relation': 176, 'relation': 2}, (3, 9): {'no relation': 144, 'relation': 1},
#               (3, 10): {'no relation': 116, 'relation': 0}, (3, 11): {'no relation': 94, 'relation': 1},
#               (3, 12): {'no relation': 73, 'relation': 2}, (3, 13): {'no relation': 65, 'relation': 0},
#               (3, 14): {'no relation': 48, 'relation': 0}, (3, 15): {'no relation': 37, 'relation': 0},
#               (4, 5): {'no relation': 258, 'relation': 73}, (4, 6): {'no relation': 257, 'relation': 12},
#               (4, 7): {'no relation': 213, 'relation': 2}, (4, 8): {'no relation': 177, 'relation': 1},
#               (4, 9): {'no relation': 145, 'relation': 0}, (4, 10): {'no relation': 116, 'relation': 0},
#               (4, 11): {'no relation': 95, 'relation': 0}, (4, 12): {'no relation': 75, 'relation': 0},
#               (4, 13): {'no relation': 66, 'relation': 0}, (4, 14): {'no relation': 47, 'relation': 0},
#               (4, 15): {'no relation': 37, 'relation': 0}, (4, 16): {'no relation': 32, 'relation': 0},
#               (5, 6): {'no relation': 208, 'relation': 61}, (5, 7): {'no relation': 199, 'relation': 16},
#               (5, 8): {'no relation': 175, 'relation': 3}, (5, 9): {'no relation': 145, 'relation': 0},
#               (5, 10): {'no relation': 116, 'relation': 0}, (5, 11): {'no relation': 95, 'relation': 0},
#               (5, 12): {'no relation': 75, 'relation': 0}, (5, 13): {'no relation': 66, 'relation': 0},
#               (5, 14): {'no relation': 48, 'relation': 0}, (5, 15): {'no relation': 37, 'relation': 0},
#               (5, 16): {'no relation': 32, 'relation': 0}, (5, 17): {'no relation': 27, 'relation': 0},
#               (6, 7): {'no relation': 172, 'relation': 43}, (6, 8): {'no relation': 170, 'relation': 8},
#               (6, 9): {'no relation': 144, 'relation': 1}, (6, 10): {'no relation': 116, 'relation': 0},
#               (6, 11): {'no relation': 94, 'relation': 1}, (6, 12): {'no relation': 75, 'relation': 0},
#               (6, 13): {'no relation': 66, 'relation': 0}, (6, 14): {'no relation': 48, 'relation': 0},
#               (6, 15): {'no relation': 37, 'relation': 0}, (6, 16): {'no relation': 32, 'relation': 0},
#               (6, 17): {'no relation': 27, 'relation': 0}, (6, 18): {'no relation': 23, 'relation': 0},
#               (7, 8): {'no relation': 143, 'relation': 35}, (7, 9): {'no relation': 137, 'relation': 8},
#               (7, 10): {'no relation': 115, 'relation': 1}, (7, 11): {'no relation': 94, 'relation': 1},
#               (7, 12): {'no relation': 75, 'relation': 0}, (7, 13): {'no relation': 66, 'relation': 0},
#               (7, 14): {'no relation': 48, 'relation': 0}, (7, 15): {'no relation': 37, 'relation': 0},
#               (7, 16): {'no relation': 32, 'relation': 0}, (7, 17): {'no relation': 27, 'relation': 0},
#               (7, 18): {'no relation': 23, 'relation': 0}, (7, 19): {'no relation': 21, 'relation': 0},
#               (8, 9): {'no relation': 118, 'relation': 27}, (8, 10): {'no relation': 109, 'relation': 7},
#               (8, 11): {'no relation': 92, 'relation': 3}, (8, 12): {'no relation': 75, 'relation': 0},
#               (8, 13): {'no relation': 66, 'relation': 0}, (8, 14): {'no relation': 48, 'relation': 0},
#               (8, 15): {'no relation': 37, 'relation': 0}, (8, 16): {'no relation': 32, 'relation': 0},
#               (8, 17): {'no relation': 27, 'relation': 0}, (8, 18): {'no relation': 23, 'relation': 0},
#               (8, 19): {'no relation': 21, 'relation': 0}, (8, 20): {'no relation': 18, 'relation': 0},
#               (9, 10): {'no relation': 93, 'relation': 23}, (9, 11): {'no relation': 86, 'relation': 9},
#               (9, 12): {'no relation': 73, 'relation': 2}, (9, 13): {'no relation': 63, 'relation': 3},
#               (9, 14): {'no relation': 48, 'relation': 0}, (9, 15): {'no relation': 37, 'relation': 0},
#               (9, 16): {'no relation': 32, 'relation': 0}, (9, 17): {'no relation': 27, 'relation': 0},
#               (9, 18): {'no relation': 23, 'relation': 0}, (9, 19): {'no relation': 21, 'relation': 0},
#               (9, 20): {'no relation': 18, 'relation': 0}, (9, 21): {'no relation': 17, 'relation': 0},
#               (10, 11): {'no relation': 68, 'relation': 27}, (10, 12): {'no relation': 73, 'relation': 2},
#               (10, 13): {'no relation': 65, 'relation': 1}, (10, 14): {'no relation': 48, 'relation': 0},
#               (10, 15): {'no relation': 37, 'relation': 0}, (10, 16): {'no relation': 32, 'relation': 0},
#               (10, 17): {'no relation': 27, 'relation': 0}, (10, 18): {'no relation': 23, 'relation': 0},
#               (10, 19): {'no relation': 21, 'relation': 0}, (10, 20): {'no relation': 18, 'relation': 0},
#               (10, 21): {'no relation': 17, 'relation': 0}, (10, 22): {'no relation': 15, 'relation': 0},
#               (11, 12): {'no relation': 61, 'relation': 14}, (11, 13): {'no relation': 66, 'relation': 0},
#               (11, 14): {'no relation': 48, 'relation': 0}, (11, 15): {'no relation': 37, 'relation': 0},
#               (11, 16): {'no relation': 32, 'relation': 0}, (11, 17): {'no relation': 27, 'relation': 0},
#               (11, 18): {'no relation': 23, 'relation': 0}, (11, 19): {'no relation': 21, 'relation': 0},
#               (11, 20): {'no relation': 18, 'relation': 0}, (11, 21): {'no relation': 17, 'relation': 0},
#               (11, 22): {'no relation': 15, 'relation': 0}, (11, 23): {'no relation': 11, 'relation': 0},
#               (12, 13): {'no relation': 52, 'relation': 14}, (12, 14): {'no relation': 47, 'relation': 1},
#               (12, 15): {'no relation': 37, 'relation': 0}, (12, 16): {'no relation': 32, 'relation': 0},
#               (12, 17): {'no relation': 27, 'relation': 0}, (12, 18): {'no relation': 23, 'relation': 0},
#               (12, 19): {'no relation': 21, 'relation': 0}, (12, 20): {'no relation': 18, 'relation': 0},
#               (12, 21): {'no relation': 17, 'relation': 0}, (12, 22): {'no relation': 15, 'relation': 0},
#               (12, 23): {'no relation': 11, 'relation': 0}, (12, 24): {'no relation': 8, 'relation': 0},
#               (13, 14): {'no relation': 43, 'relation': 5}, (13, 15): {'no relation': 36, 'relation': 1},
#               (13, 16): {'no relation': 32, 'relation': 0}, (13, 17): {'no relation': 27, 'relation': 0},
#               (13, 18): {'no relation': 23, 'relation': 0}, (13, 19): {'no relation': 21, 'relation': 0},
#               (13, 20): {'no relation': 18, 'relation': 0}, (13, 21): {'no relation': 17, 'relation': 0},
#               (13, 22): {'no relation': 15, 'relation': 0}, (13, 23): {'no relation': 11, 'relation': 0},
#               (13, 24): {'no relation': 8, 'relation': 0}, (13, 25): {'no relation': 6, 'relation': 0},
#               (14, 15): {'no relation': 31, 'relation': 6}, (14, 16): {'no relation': 32, 'relation': 0},
#               (14, 17): {'no relation': 27, 'relation': 0}, (14, 18): {'no relation': 23, 'relation': 0},
#               (14, 19): {'no relation': 21, 'relation': 0}, (14, 20): {'no relation': 18, 'relation': 0},
#               (14, 21): {'no relation': 17, 'relation': 0}, (14, 22): {'no relation': 15, 'relation': 0},
#               (14, 23): {'no relation': 11, 'relation': 0}, (14, 24): {'no relation': 8, 'relation': 0},
#               (14, 25): {'no relation': 6, 'relation': 0}, (14, 26): {'no relation': 5, 'relation': 0},
#               (15, 16): {'no relation': 28, 'relation': 4}, (15, 17): {'no relation': 27, 'relation': 0},
#               (15, 18): {'no relation': 23, 'relation': 0}, (15, 19): {'no relation': 21, 'relation': 0},
#               (15, 20): {'no relation': 18, 'relation': 0}, (15, 21): {'no relation': 17, 'relation': 0},
#               (15, 22): {'no relation': 15, 'relation': 0}, (15, 23): {'no relation': 11, 'relation': 0},
#               (15, 24): {'no relation': 8, 'relation': 0}, (15, 25): {'no relation': 6, 'relation': 0},
#               (15, 26): {'no relation': 5, 'relation': 0}, (15, 27): {'no relation': 3, 'relation': 0},
#               (16, 17): {'no relation': 20, 'relation': 7}, (16, 18): {'no relation': 23, 'relation': 0},
#               (16, 19): {'no relation': 21, 'relation': 0}, (16, 20): {'no relation': 18, 'relation': 0},
#               (16, 21): {'no relation': 17, 'relation': 0}, (16, 22): {'no relation': 15, 'relation': 0},
#               (16, 23): {'no relation': 11, 'relation': 0}, (16, 24): {'no relation': 8, 'relation': 0},
#               (16, 25): {'no relation': 6, 'relation': 0}, (16, 26): {'no relation': 5, 'relation': 0},
#               (16, 27): {'no relation': 3, 'relation': 0}, (17, 18): {'no relation': 21, 'relation': 2},
#               (17, 19): {'no relation': 21, 'relation': 0}, (17, 20): {'no relation': 18, 'relation': 0},
#               (17, 21): {'no relation': 17, 'relation': 0}, (17, 22): {'no relation': 15, 'relation': 0},
#               (17, 23): {'no relation': 11, 'relation': 0}, (17, 24): {'no relation': 8, 'relation': 0},
#               (17, 25): {'no relation': 6, 'relation': 0}, (17, 26): {'no relation': 5, 'relation': 0},
#               (17, 27): {'no relation': 3, 'relation': 0}, (18, 19): {'no relation': 17, 'relation': 4},
#               (18, 20): {'no relation': 18, 'relation': 0}, (18, 21): {'no relation': 17, 'relation': 0},
#               (18, 22): {'no relation': 15, 'relation': 0}, (18, 23): {'no relation': 11, 'relation': 0},
#               (18, 24): {'no relation': 8, 'relation': 0}, (18, 25): {'no relation': 6, 'relation': 0},
#               (18, 26): {'no relation': 5, 'relation': 0}, (18, 27): {'no relation': 3, 'relation': 0},
#               (19, 20): {'no relation': 12, 'relation': 6}, (19, 21): {'no relation': 15, 'relation': 2},
#               (19, 22): {'no relation': 15, 'relation': 0}, (19, 23): {'no relation': 11, 'relation': 0},
#               (19, 24): {'no relation': 8, 'relation': 0}, (19, 25): {'no relation': 6, 'relation': 0},
#               (19, 26): {'no relation': 5, 'relation': 0}, (19, 27): {'no relation': 3, 'relation': 0},
#               (20, 21): {'no relation': 14, 'relation': 3}, (20, 22): {'no relation': 15, 'relation': 0},
#               (20, 23): {'no relation': 11, 'relation': 0}, (20, 24): {'no relation': 8, 'relation': 0},
#               (20, 25): {'no relation': 6, 'relation': 0}, (20, 26): {'no relation': 5, 'relation': 0},
#               (20, 27): {'no relation': 3, 'relation': 0}, (21, 22): {'no relation': 14, 'relation': 1},
#               (21, 23): {'no relation': 10, 'relation': 1}, (21, 24): {'no relation': 8, 'relation': 0},
#               (21, 25): {'no relation': 6, 'relation': 0}, (21, 26): {'no relation': 5, 'relation': 0},
#               (21, 27): {'no relation': 3, 'relation': 0}, (22, 23): {'no relation': 9, 'relation': 2},
#               (22, 24): {'no relation': 7, 'relation': 1}, (22, 25): {'no relation': 6, 'relation': 0},
#               (22, 26): {'no relation': 5, 'relation': 0}, (22, 27): {'no relation': 3, 'relation': 0},
#               (23, 24): {'no relation': 7, 'relation': 1}, (23, 25): {'no relation': 6, 'relation': 0},
#               (23, 26): {'no relation': 4, 'relation': 1}, (23, 27): {'no relation': 3, 'relation': 0},
#               (24, 25): {'no relation': 5, 'relation': 1}, (24, 26): {'no relation': 5, 'relation': 0},
#               (24, 27): {'no relation': 3, 'relation': 0}, (25, 26): {'no relation': 2, 'relation': 3},
#               (25, 27): {'no relation': 3, 'relation': 0}, (26, 27): {'no relation': 3, 'relation': 0}}
#
# ar_idx_p_map = {}
# for key, value in ar_idx_map.items():
#     n1, n2 = value["no relation"], value["relation"]
#     n = n1 + n2
#     ar_idx_p_map[key] = tuple([n1 / n, n2 / n])
# print("ar_idx_p_map", ar_idx_p_map)
#
# ar_type_idx_map = {(0, 1): {'reason': 226, 'evidence': 5}, (0, 2): {'reason': 89, 'evidence': 4},
#                    (0, 3): {'reason': 39, 'evidence': 1}, (0, 4): {'reason': 18, 'evidence': 1},
#                    (0, 5): {'reason': 7, 'evidence': 0}, (0, 6): {'reason': 1, 'evidence': 1},
#                    (0, 7): {'reason': 4, 'evidence': 0}, (0, 8): {'reason': 4, 'evidence': 0},
#                    (0, 9): {'reason': 2, 'evidence': 0}, (0, 10): {'reason': 0, 'evidence': 0},
#                    (0, 11): {'reason': 0, 'evidence': 0}, (0, 12): {'reason': 0, 'evidence': 0},
#                    (1, 2): {'reason': 118, 'evidence': 5}, (1, 3): {'reason': 26, 'evidence': 3},
#                    (1, 4): {'reason': 14, 'evidence': 1}, (1, 5): {'reason': 4, 'evidence': 0},
#                    (1, 6): {'reason': 1, 'evidence': 1}, (1, 7): {'reason': 2, 'evidence': 0},
#                    (1, 8): {'reason': 1, 'evidence': 0}, (1, 9): {'reason': 1, 'evidence': 0},
#                    (1, 10): {'reason': 0, 'evidence': 0}, (1, 11): {'reason': 0, 'evidence': 0},
#                    (1, 12): {'reason': 0, 'evidence': 0}, (1, 13): {'reason': 0, 'evidence': 0},
#                    (2, 3): {'reason': 123, 'evidence': 5}, (2, 4): {'reason': 34, 'evidence': 1},
#                    (2, 5): {'reason': 7, 'evidence': 2}, (2, 6): {'reason': 4, 'evidence': 1},
#                    (2, 7): {'reason': 2, 'evidence': 0}, (2, 8): {'reason': 0, 'evidence': 0},
#                    (2, 9): {'reason': 0, 'evidence': 0}, (2, 10): {'reason': 0, 'evidence': 0},
#                    (2, 11): {'reason': 0, 'evidence': 0}, (2, 12): {'reason': 0, 'evidence': 0},
#                    (2, 13): {'reason': 0, 'evidence': 0}, (2, 14): {'reason': 0, 'evidence': 0},
#                    (3, 4): {'reason': 95, 'evidence': 3}, (3, 5): {'reason': 23, 'evidence': 0},
#                    (3, 6): {'reason': 7, 'evidence': 1}, (3, 7): {'reason': 3, 'evidence': 0},
#                    (3, 8): {'reason': 2, 'evidence': 0}, (3, 9): {'reason': 1, 'evidence': 0},
#                    (3, 10): {'reason': 0, 'evidence': 0}, (3, 11): {'reason': 1, 'evidence': 0},
#                    (3, 12): {'reason': 1, 'evidence': 1}, (3, 13): {'reason': 0, 'evidence': 0},
#                    (3, 14): {'reason': 0, 'evidence': 0}, (3, 15): {'reason': 0, 'evidence': 0},
#                    (4, 5): {'reason': 72, 'evidence': 1}, (4, 6): {'reason': 11, 'evidence': 1},
#                    (4, 7): {'reason': 2, 'evidence': 0}, (4, 8): {'reason': 1, 'evidence': 0},
#                    (4, 9): {'reason': 0, 'evidence': 0}, (4, 10): {'reason': 0, 'evidence': 0},
#                    (4, 11): {'reason': 0, 'evidence': 0}, (4, 12): {'reason': 0, 'evidence': 0},
#                    (4, 13): {'reason': 0, 'evidence': 0}, (4, 14): {'reason': 0, 'evidence': 0},
#                    (4, 15): {'reason': 0, 'evidence': 0}, (4, 16): {'reason': 0, 'evidence': 0},
#                    (5, 6): {'reason': 61, 'evidence': 0}, (5, 7): {'reason': 16, 'evidence': 0},
#                    (5, 8): {'reason': 3, 'evidence': 0}, (5, 9): {'reason': 0, 'evidence': 0},
#                    (5, 10): {'reason': 0, 'evidence': 0}, (5, 11): {'reason': 0, 'evidence': 0},
#                    (5, 12): {'reason': 0, 'evidence': 0}, (5, 13): {'reason': 0, 'evidence': 0},
#                    (5, 14): {'reason': 0, 'evidence': 0}, (5, 15): {'reason': 0, 'evidence': 0},
#                    (5, 16): {'reason': 0, 'evidence': 0}, (5, 17): {'reason': 0, 'evidence': 0},
#                    (6, 7): {'reason': 42, 'evidence': 1}, (6, 8): {'reason': 8, 'evidence': 0},
#                    (6, 9): {'reason': 1, 'evidence': 0}, (6, 10): {'reason': 0, 'evidence': 0},
#                    (6, 11): {'reason': 1, 'evidence': 0}, (6, 12): {'reason': 0, 'evidence': 0},
#                    (6, 13): {'reason': 0, 'evidence': 0}, (6, 14): {'reason': 0, 'evidence': 0},
#                    (6, 15): {'reason': 0, 'evidence': 0}, (6, 16): {'reason': 0, 'evidence': 0},
#                    (6, 17): {'reason': 0, 'evidence': 0}, (6, 18): {'reason': 0, 'evidence': 0},
#                    (7, 8): {'reason': 34, 'evidence': 1}, (7, 9): {'reason': 8, 'evidence': 0},
#                    (7, 10): {'reason': 1, 'evidence': 0}, (7, 11): {'reason': 1, 'evidence': 0},
#                    (7, 12): {'reason': 0, 'evidence': 0}, (7, 13): {'reason': 0, 'evidence': 0},
#                    (7, 14): {'reason': 0, 'evidence': 0}, (7, 15): {'reason': 0, 'evidence': 0},
#                    (7, 16): {'reason': 0, 'evidence': 0}, (7, 17): {'reason': 0, 'evidence': 0},
#                    (7, 18): {'reason': 0, 'evidence': 0}, (7, 19): {'reason': 0, 'evidence': 0},
#                    (8, 9): {'reason': 27, 'evidence': 0}, (8, 10): {'reason': 7, 'evidence': 0},
#                    (8, 11): {'reason': 3, 'evidence': 0}, (8, 12): {'reason': 0, 'evidence': 0},
#                    (8, 13): {'reason': 0, 'evidence': 0}, (8, 14): {'reason': 0, 'evidence': 0},
#                    (8, 15): {'reason': 0, 'evidence': 0}, (8, 16): {'reason': 0, 'evidence': 0},
#                    (8, 17): {'reason': 0, 'evidence': 0}, (8, 18): {'reason': 0, 'evidence': 0},
#                    (8, 19): {'reason': 0, 'evidence': 0}, (8, 20): {'reason': 0, 'evidence': 0},
#                    (9, 10): {'reason': 23, 'evidence': 0}, (9, 11): {'reason': 9, 'evidence': 0},
#                    (9, 12): {'reason': 2, 'evidence': 0}, (9, 13): {'reason': 3, 'evidence': 0},
#                    (9, 14): {'reason': 0, 'evidence': 0}, (9, 15): {'reason': 0, 'evidence': 0},
#                    (9, 16): {'reason': 0, 'evidence': 0}, (9, 17): {'reason': 0, 'evidence': 0},
#                    (9, 18): {'reason': 0, 'evidence': 0}, (9, 19): {'reason': 0, 'evidence': 0},
#                    (9, 20): {'reason': 0, 'evidence': 0}, (9, 21): {'reason': 0, 'evidence': 0},
#                    (10, 11): {'reason': 27, 'evidence': 0}, (10, 12): {'reason': 2, 'evidence': 0},
#                    (10, 13): {'reason': 1, 'evidence': 0}, (10, 14): {'reason': 0, 'evidence': 0},
#                    (10, 15): {'reason': 0, 'evidence': 0}, (10, 16): {'reason': 0, 'evidence': 0},
#                    (10, 17): {'reason': 0, 'evidence': 0}, (10, 18): {'reason': 0, 'evidence': 0},
#                    (10, 19): {'reason': 0, 'evidence': 0}, (10, 20): {'reason': 0, 'evidence': 0},
#                    (10, 21): {'reason': 0, 'evidence': 0}, (10, 22): {'reason': 0, 'evidence': 0},
#                    (11, 12): {'reason': 14, 'evidence': 0}, (11, 13): {'reason': 0, 'evidence': 0},
#                    (11, 14): {'reason': 0, 'evidence': 0}, (11, 15): {'reason': 0, 'evidence': 0},
#                    (11, 16): {'reason': 0, 'evidence': 0}, (11, 17): {'reason': 0, 'evidence': 0},
#                    (11, 18): {'reason': 0, 'evidence': 0}, (11, 19): {'reason': 0, 'evidence': 0},
#                    (11, 20): {'reason': 0, 'evidence': 0}, (11, 21): {'reason': 0, 'evidence': 0},
#                    (11, 22): {'reason': 0, 'evidence': 0}, (11, 23): {'reason': 0, 'evidence': 0},
#                    (12, 13): {'reason': 14, 'evidence': 0}, (12, 14): {'reason': 1, 'evidence': 0},
#                    (12, 15): {'reason': 0, 'evidence': 0}, (12, 16): {'reason': 0, 'evidence': 0},
#                    (12, 17): {'reason': 0, 'evidence': 0}, (12, 18): {'reason': 0, 'evidence': 0},
#                    (12, 19): {'reason': 0, 'evidence': 0}, (12, 20): {'reason': 0, 'evidence': 0},
#                    (12, 21): {'reason': 0, 'evidence': 0}, (12, 22): {'reason': 0, 'evidence': 0},
#                    (12, 23): {'reason': 0, 'evidence': 0}, (12, 24): {'reason': 0, 'evidence': 0},
#                    (13, 14): {'reason': 3, 'evidence': 2}, (13, 15): {'reason': 0, 'evidence': 1},
#                    (13, 16): {'reason': 0, 'evidence': 0}, (13, 17): {'reason': 0, 'evidence': 0},
#                    (13, 18): {'reason': 0, 'evidence': 0}, (13, 19): {'reason': 0, 'evidence': 0},
#                    (13, 20): {'reason': 0, 'evidence': 0}, (13, 21): {'reason': 0, 'evidence': 0},
#                    (13, 22): {'reason': 0, 'evidence': 0}, (13, 23): {'reason': 0, 'evidence': 0},
#                    (13, 24): {'reason': 0, 'evidence': 0}, (13, 25): {'reason': 0, 'evidence': 0},
#                    (14, 15): {'reason': 5, 'evidence': 1}, (14, 16): {'reason': 0, 'evidence': 0},
#                    (14, 17): {'reason': 0, 'evidence': 0}, (14, 18): {'reason': 0, 'evidence': 0},
#                    (14, 19): {'reason': 0, 'evidence': 0}, (14, 20): {'reason': 0, 'evidence': 0},
#                    (14, 21): {'reason': 0, 'evidence': 0}, (14, 22): {'reason': 0, 'evidence': 0},
#                    (14, 23): {'reason': 0, 'evidence': 0}, (14, 24): {'reason': 0, 'evidence': 0},
#                    (14, 25): {'reason': 0, 'evidence': 0}, (14, 26): {'reason': 0, 'evidence': 0},
#                    (15, 16): {'reason': 4, 'evidence': 0}, (15, 17): {'reason': 0, 'evidence': 0},
#                    (15, 18): {'reason': 0, 'evidence': 0}, (15, 19): {'reason': 0, 'evidence': 0},
#                    (15, 20): {'reason': 0, 'evidence': 0}, (15, 21): {'reason': 0, 'evidence': 0},
#                    (15, 22): {'reason': 0, 'evidence': 0}, (15, 23): {'reason': 0, 'evidence': 0},
#                    (15, 24): {'reason': 0, 'evidence': 0}, (15, 25): {'reason': 0, 'evidence': 0},
#                    (15, 26): {'reason': 0, 'evidence': 0}, (15, 27): {'reason': 0, 'evidence': 0},
#                    (16, 17): {'reason': 6, 'evidence': 1}, (16, 18): {'reason': 0, 'evidence': 0},
#                    (16, 19): {'reason': 0, 'evidence': 0}, (16, 20): {'reason': 0, 'evidence': 0},
#                    (16, 21): {'reason': 0, 'evidence': 0}, (16, 22): {'reason': 0, 'evidence': 0},
#                    (16, 23): {'reason': 0, 'evidence': 0}, (16, 24): {'reason': 0, 'evidence': 0},
#                    (16, 25): {'reason': 0, 'evidence': 0}, (16, 26): {'reason': 0, 'evidence': 0},
#                    (16, 27): {'reason': 0, 'evidence': 0}, (17, 18): {'reason': 2, 'evidence': 0},
#                    (17, 19): {'reason': 0, 'evidence': 0}, (17, 20): {'reason': 0, 'evidence': 0},
#                    (17, 21): {'reason': 0, 'evidence': 0}, (17, 22): {'reason': 0, 'evidence': 0},
#                    (17, 23): {'reason': 0, 'evidence': 0}, (17, 24): {'reason': 0, 'evidence': 0},
#                    (17, 25): {'reason': 0, 'evidence': 0}, (17, 26): {'reason': 0, 'evidence': 0},
#                    (17, 27): {'reason': 0, 'evidence': 0}, (18, 19): {'reason': 4, 'evidence': 0},
#                    (18, 20): {'reason': 0, 'evidence': 0}, (18, 21): {'reason': 0, 'evidence': 0},
#                    (18, 22): {'reason': 0, 'evidence': 0}, (18, 23): {'reason': 0, 'evidence': 0},
#                    (18, 24): {'reason': 0, 'evidence': 0}, (18, 25): {'reason': 0, 'evidence': 0},
#                    (18, 26): {'reason': 0, 'evidence': 0}, (18, 27): {'reason': 0, 'evidence': 0},
#                    (19, 20): {'reason': 6, 'evidence': 0}, (19, 21): {'reason': 2, 'evidence': 0},
#                    (19, 22): {'reason': 0, 'evidence': 0}, (19, 23): {'reason': 0, 'evidence': 0},
#                    (19, 24): {'reason': 0, 'evidence': 0}, (19, 25): {'reason': 0, 'evidence': 0},
#                    (19, 26): {'reason': 0, 'evidence': 0}, (19, 27): {'reason': 0, 'evidence': 0},
#                    (20, 21): {'reason': 3, 'evidence': 0}, (20, 22): {'reason': 0, 'evidence': 0},
#                    (20, 23): {'reason': 0, 'evidence': 0}, (20, 24): {'reason': 0, 'evidence': 0},
#                    (20, 25): {'reason': 0, 'evidence': 0}, (20, 26): {'reason': 0, 'evidence': 0},
#                    (20, 27): {'reason': 0, 'evidence': 0}, (21, 22): {'reason': 1, 'evidence': 0},
#                    (21, 23): {'reason': 1, 'evidence': 0}, (21, 24): {'reason': 0, 'evidence': 0},
#                    (21, 25): {'reason': 0, 'evidence': 0}, (21, 26): {'reason': 0, 'evidence': 0},
#                    (21, 27): {'reason': 0, 'evidence': 0}, (22, 23): {'reason': 2, 'evidence': 0},
#                    (22, 24): {'reason': 1, 'evidence': 0}, (22, 25): {'reason': 0, 'evidence': 0},
#                    (22, 26): {'reason': 0, 'evidence': 0}, (22, 27): {'reason': 0, 'evidence': 0},
#                    (23, 24): {'reason': 1, 'evidence': 0}, (23, 25): {'reason': 0, 'evidence': 0},
#                    (23, 26): {'reason': 1, 'evidence': 0}, (23, 27): {'reason': 0, 'evidence': 0},
#                    (24, 25): {'reason': 1, 'evidence': 0}, (24, 26): {'reason': 0, 'evidence': 0},
#                    (24, 27): {'reason': 0, 'evidence': 0}, (25, 26): {'reason': 3, 'evidence': 0},
#                    (25, 27): {'reason': 0, 'evidence': 0}, (26, 27): {'reason': 0, 'evidence': 0}}
#
# ar_type_idx_p_map = {}
# for key, value in ar_type_idx_map.items():
#     n1, n2 = value["reason"], value["evidence"]
#     n = n1 + n2
#     if n1 ==0 and n2 == 0:
#         n1 = 1
#         n2 = 1
#         n = 2
#     ar_type_idx_p_map[key] = tuple([n1 / n, n2 / n])
# print("ar_type_idx_p_map", ar_type_idx_p_map)



# with open(f"./dataset/{model_name_or_path}_{dataset_name}.pt", "wb") as file:
#     torch.save(t, file)