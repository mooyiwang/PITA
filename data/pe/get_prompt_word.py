from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json
import pandas as pd
import random
from models.pos_map import pair2sequence, bart_prefix_ac_map, pair_idx_map


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

data_df = pd.read_csv("pe_data_df.csv")

split_test_file_path = "./data/pe/test_paragraph_index.json"
with open(split_test_file_path, "r") as fp:
    test_id_list = json.load(fp)
test_data_df = data_df[data_df["para_id"].isin(test_id_list)]
train_data_df = data_df[~(data_df["para_id"].isin(test_id_list))]

essay_id2parag_id_dict = train_data_df.groupby("essay_id").groups
essay_id_list = list(essay_id2parag_id_dict.keys())
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

    ac_type_map = {"MajorClaim": 0, "Claim": 0, "Premise": 0}
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

    ar_type_map = {"Support": 0, "Attack": 0}
    for i, AR_types in enumerate(AR_types_list):
        ar_num = len(AR_types)
        if ar_num == 0:
            continue
        for type in AR_types:
            ar_type_map[type] += 1

    print("ac_type_map", ac_type_map)
    print("ar_map", ar_map)
    print("ar_type_map", ar_type_map)

get_label_dsitribution(train_data_df)
get_label_dsitribution(dev_data_df)
get_label_dsitribution(test_data_df)




# data_df = pd.read_csv("pe_data_df.csv")
# AC_types_list = [list(eval(AC_types)) for AC_types in data_df["ac_types"]]
# AR_pairs_list = [eval(_) for _ in data_df['ac_rel_pairs']] # ac_types,ac_rel_targets,ac_rel_types,ac_rel_pairs
# AR_types_list = [eval(_) for _ in data_df['ac_rel_types']]

# ac_type_map = {'MajorClaim': 0, 'Claim': 0, 'Premise': 0}
# for AC_types in AC_types_list:
#     if len(AC_types) == 0:
#         continue
#     for type in AC_types:
#         ac_type_map[type] += 1
#
# ar_map = {'no relation':0 , 'relation': 0}
# for i, AR_pairs in enumerate(AR_pairs_list):
#     ac_num = len(AC_types_list[i])
#     if ac_num == 1 or ac_num == 0:
#         continue
#     rel_num = len(AR_pairs)
#     # print("ac_num", ac_num)
#     no_rel_num = len(pair_idx_map[ac_num]) - rel_num
#     ar_map['no relation'] += no_rel_num
#     ar_map['relation'] += rel_num
#
# ar_type_map = {'Support':0 , 'Attack': 0}
# for i, AR_types in enumerate(AR_type_list):
#     ar_num = len(AR_types)
#     if ar_num == 0:
#         continue
#     for type in AR_types:
#         ar_type_map[type] += 1
#
# print("ac_type_map", ac_type_map)
# print("ar_map", ar_map)
# print("ar_type_map", ar_type_map)
# ac_type_map {'MajorClaim': 751, 'Claim': 1506, 'Premise': 3832}
# ar_map {'no relation': 7248, 'relation': 3832}
# ar_type_map {'Support': 3613, 'Attack': 219}


# ac_type_idx_map = { i : {'MajorClaim': 0, 'Claim': 0, 'Premise': 0} for i in range(12)}
# ar_idx_map = {pair: {'no relation':0 , 'relation': 0} for pair in list(pair2sequence[12])}
# ar_type_idx_map = {pair: {'Support':0 , 'Attack': 0} for pair in list(pair2sequence[12])}
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

ac_type_idx_map = {
0: {'MajorClaim': 579, 'Claim': 788, 'Premise': 397}, 1: {'MajorClaim': 129, 'Claim': 296, 'Premise': 963},
2: {'MajorClaim': 26, 'Claim': 143, 'Premise': 905}, 3: {'MajorClaim': 11, 'Claim': 117, 'Premise': 689},
4: {'MajorClaim': 4, 'Claim': 78, 'Premise': 437}, 5: {'MajorClaim': 2, 'Claim': 47, 'Premise': 243},
6: {'MajorClaim': 0, 'Claim': 25, 'Premise': 116}, 7: {'MajorClaim': 0, 'Claim': 5, 'Premise': 51},
8: {'MajorClaim': 0, 'Claim': 6, 'Premise': 17}, 9: {'MajorClaim': 0, 'Claim': 1, 'Premise': 9},
10: {'MajorClaim': 0, 'Claim': 0, 'Premise': 4}, 11: {'MajorClaim': 0, 'Claim': 0, 'Premise': 1}
}
ac_type_idx_p_map = {}
for key, value in ac_type_idx_map.items():
    n1, n2, n3 = value["MajorClaim"], value["Claim"], value["Premise"]
    n = n1 + n2 + n3
    ac_type_idx_p_map[key] = tuple([n1 / n, n2 / n, n3 / n])
print("ac_type_idx_p_map", ac_type_idx_p_map)


ar_idx_map = {
(0, 1): {'no relation': 642, 'relation': 746}, (0, 2): {'no relation': 518, 'relation': 556},
(0, 3): {'no relation': 384, 'relation': 433}, (0, 4): {'no relation': 274, 'relation': 245},
(0, 5): {'no relation': 148, 'relation': 144}, (0, 6): {'no relation': 90, 'relation': 51},
(0, 7): {'no relation': 35, 'relation': 21}, (0, 8): {'no relation': 15, 'relation': 8},
(0, 9): {'no relation': 7, 'relation': 3}, (1, 2): {'no relation': 746, 'relation': 328},
(1, 3): {'no relation': 677, 'relation': 140}, (1, 4): {'no relation': 445, 'relation': 74},
(1, 5): {'no relation': 248, 'relation': 44}, (1, 6): {'no relation': 118, 'relation': 23},
(1, 7): {'no relation': 46, 'relation': 10}, (1, 8): {'no relation': 20, 'relation': 3},
(1, 9): {'no relation': 9, 'relation': 1}, (1, 10): {'no relation': 4, 'relation': 0},
(2, 3): {'no relation': 535, 'relation': 282}, (2, 4): {'no relation': 428, 'relation': 91},
(2, 5): {'no relation': 247, 'relation': 45}, (2, 6): {'no relation': 124, 'relation': 17},
(2, 7): {'no relation': 48, 'relation': 8}, (2, 8): {'no relation': 21, 'relation': 2},
(2, 9): {'no relation': 8, 'relation': 2}, (2, 10): {'no relation': 3, 'relation': 1},
(2, 11): {'no relation': 1, 'relation': 0}, (3, 4): {'no relation': 318, 'relation': 201},
(3, 5): {'no relation': 238, 'relation': 54}, (3, 6): {'no relation': 120, 'relation': 21},
(3, 7): {'no relation': 48, 'relation': 8}, (3, 8): {'no relation': 20, 'relation': 3},
(3, 9): {'no relation': 8, 'relation': 2}, (3, 10): {'no relation': 4, 'relation': 0},
(3, 11): {'no relation': 1, 'relation': 0}, (4, 5): {'no relation': 181, 'relation': 111},
(4, 6): {'no relation': 109, 'relation': 32}, (4, 7): {'no relation': 48, 'relation': 8},
(4, 8): {'no relation': 21, 'relation': 2}, (4, 9): {'no relation': 10, 'relation': 0},
(4, 10): {'no relation': 4, 'relation': 0}, (4, 11): {'no relation': 1, 'relation': 0},
(5, 6): {'no relation': 90, 'relation': 51}, (5, 7): {'no relation': 46, 'relation': 10},
(5, 8): {'no relation': 20, 'relation': 3}, (5, 9): {'no relation': 10, 'relation': 0},
(5, 10): {'no relation': 4, 'relation': 0}, (5, 11): {'no relation': 1, 'relation': 0},
(6, 7): {'no relation': 34, 'relation': 22}, (6, 8): {'no relation': 18, 'relation': 5},
(6, 9): {'no relation': 8, 'relation': 2}, (6, 10): {'no relation': 4, 'relation': 0},
(6, 11): {'no relation': 1, 'relation': 0}, (7, 8): {'no relation': 13, 'relation': 10},
(7, 9): {'no relation': 10, 'relation': 0}, (7, 10): {'no relation': 4, 'relation': 0},
(7, 11): {'no relation': 1, 'relation': 0}, (8, 9): {'no relation': 5, 'relation': 5},
(8, 10): {'no relation': 4, 'relation': 0}, (8, 11): {'no relation': 1, 'relation': 0},
(9, 10): {'no relation': 2, 'relation': 2}, (9, 11): {'no relation': 1, 'relation': 0},
(10, 11): {'no relation': 1, 'relation': 0}
}
ar_idx_p_map = {}
for key, value in ar_idx_map.items():
    n1, n2 = value["no relation"], value["relation"]
    n = n1 + n2
    ar_idx_p_map[key] = tuple([n1 / n, n2 / n])
print("ar_idx_p_map", ar_idx_p_map)


ar_type_idx_map = {
(0, 1): {'Support': 693, 'Attack': 53}, (0, 2): {'Support': 514, 'Attack': 42},
(0, 3): {'Support': 408, 'Attack': 25}, (0, 4): {'Support': 235, 'Attack': 10},
(0, 5): {'Support': 135, 'Attack': 9}, (0, 6): {'Support': 50, 'Attack': 1},
(0, 7): {'Support': 20, 'Attack': 1}, (0, 8): {'Support': 7, 'Attack': 1},
(0, 9): {'Support': 3, 'Attack': 0}, (1, 2): {'Support': 297, 'Attack': 31},
(1, 3): {'Support': 137, 'Attack': 3}, (1, 4): {'Support': 74, 'Attack': 0},
(1, 5): {'Support': 41, 'Attack': 3}, (1, 6): {'Support': 21, 'Attack': 2},
(1, 7): {'Support': 9, 'Attack': 1}, (1, 8): {'Support': 3, 'Attack': 0},
(1, 9): {'Support': 1, 'Attack': 0}, (1, 10): {'Support': 0, 'Attack': 0},
(2, 3): {'Support': 268, 'Attack': 14}, (2, 4): {'Support': 88, 'Attack': 3},
(2, 5): {'Support': 44, 'Attack': 1}, (2, 6): {'Support': 17, 'Attack': 0},
(2, 7): {'Support': 8, 'Attack': 0}, (2, 8): {'Support': 2, 'Attack': 0},
(2, 9): {'Support': 2, 'Attack': 0}, (2, 10): {'Support': 1, 'Attack': 0},
(2, 11): {'Support': 0, 'Attack': 0}, (3, 4): {'Support': 194, 'Attack': 7},
(3, 5): {'Support': 53, 'Attack': 1}, (3, 6): {'Support': 20, 'Attack': 1},
(3, 7): {'Support': 7, 'Attack': 1}, (3, 8): {'Support': 3, 'Attack': 0},
(3, 9): {'Support': 1, 'Attack': 1}, (3, 10): {'Support': 0, 'Attack': 0},
(3, 11): {'Support': 0, 'Attack': 0}, (4, 5): {'Support': 109, 'Attack': 2},
(4, 6): {'Support': 32, 'Attack': 0}, (4, 7): {'Support': 8, 'Attack': 0},
(4, 8): {'Support': 2, 'Attack': 0}, (4, 9): {'Support': 0, 'Attack': 0},
(4, 10): {'Support': 0, 'Attack': 0}, (4, 11): {'Support': 0, 'Attack': 0},
(5, 6): {'Support': 48, 'Attack': 3}, (5, 7): {'Support': 10, 'Attack': 0},
(5, 8): {'Support': 3, 'Attack': 0}, (5, 9): {'Support': 0, 'Attack': 0},
(5, 10): {'Support': 0, 'Attack': 0}, (5, 11): {'Support': 0, 'Attack': 0},
(6, 7): {'Support': 22, 'Attack': 0}, (6, 8): {'Support': 5, 'Attack': 0},
(6, 9): {'Support': 2, 'Attack': 0}, (6, 10): {'Support': 0, 'Attack': 0},
(6, 11): {'Support': 0, 'Attack': 0}, (7, 8): {'Support': 8, 'Attack': 2},
(7, 9): {'Support': 0, 'Attack': 0}, (7, 10): {'Support': 0, 'Attack': 0},
(7, 11): {'Support': 0, 'Attack': 0}, (8, 9): {'Support': 5, 'Attack': 0},
(8, 10): {'Support': 0, 'Attack': 0}, (8, 11): {'Support': 0, 'Attack': 0},
(9, 10): {'Support': 1, 'Attack': 1}, (9, 11): {'Support': 0, 'Attack': 0},
(10, 11): {'Support': 0, 'Attack': 0}
}
ar_type_idx_p_map = {}
for key, value in ar_type_idx_map.items():
    n1, n2 = value["Support"], value["Attack"]
    n = n1 + n2
    if n1 ==0 and n2 == 0:
        n1 = 1
        n2 = 1
        n = 2
    ar_type_idx_p_map[key] = tuple([n1 / n, n2 / n])
print("ar_type_idx_p_map", ar_type_idx_p_map)




# with open(f"./dataset/{model_name_or_path}_{dataset_name}.pt", "wb") as file:
#     torch.save(t, file)