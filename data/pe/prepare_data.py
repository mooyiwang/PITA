# encoding: utf-8
# @author: 
# email: 

import json
import numpy as np
import os, sys
from collections import defaultdict
import itertools
import datetime
import pandas as pd
import json
import random
import argparse
from collections import Counter


def get_data_dicts(data):

    SCRIPT_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(SCRIPT_PATH, data)

    essay_info_dict, essay_max_n_dict, para_info_dict = get_essay_info_dict(DATA_PATH)

    return essay_info_dict, essay_max_n_dict, para_info_dict


def get_essay_info_dict(FILENAME):

    with open(FILENAME) as f:
        n_span_para = [] # AC_id_in_paragraph
        n_span_essay = [] # AC_id_in_essay
        n_para = [] # Paragraph_id_in_essay
        for line in f:
            if line.split("\t")[6] != "-" \
                and line.split("\t")[5] != "AC_id_in_essay" \
                and line.split("\t")[6] != "AC_id_in_paragraph":
                n_span_essay.append(int(line.split("\t")[5]))
                n_span_para.append(int(line.split("\t")[6]))
                n_para.append(int(line.split("\t")[2]))

    max_n_spans = max(n_span_para) + 1
    max_n_paras = max(n_para) + 1

    essay_info_dict = {}
    split_column = 1

    essay2parainfo = defaultdict(dict)
    essay2paraids = defaultdict(list)
    para2essayid = dict()
    
    # idx = 0
    with open(FILENAME) as f:
        # for every paragraph; groupby can group words into group
        for essay_id, lines in itertools.groupby(f, key=lambda x: x.split("\t")[split_column]):
            if essay_id == "Essay_id" or essay_id == "Paragraph_id" or essay_id == "-":
                continue

            essay_lines = list(lines)
            para_type = essay_lines[0].split("\t")[3]
            essay_id = int(essay_lines[0].split("\t")[0])
            para_id = int(essay_lines[0].split("\t")[1])

            para2essayid[para_id] = essay_id
            essay2paraids[essay_id].append(para_id)
            essay2parainfo[essay_id][para_type] = para_id

            essay_info_dict[int(para_id)] = get_essay_detail(essay_lines,
                                                             max_n_spans)
            # idx += 1
            # if idx == 10:
            #     break

    max_n_tokens = max([len(essay_info_dict[essay_id]["text"])
                        for essay_id in range(len(essay_info_dict))])

    essay_max_n_dict = {}
    essay_max_n_dict["max_n_spans_para"] = max_n_spans
    essay_max_n_dict["max_n_paras"] = max_n_paras
    essay_max_n_dict["max_n_tokens"] = max_n_tokens
    # for each para_id(key), the structure info of the essay it in(value)
    para_info_dict = defaultdict(dict)
    for para_id, essay_id in para2essayid.items():
        para_info_dict[para_id]["prompt"] = essay2parainfo[essay_id]["prompt"]
        para_info_dict[para_id]["intro"] = essay2parainfo[essay_id]["intro"]
        para_info_dict[para_id]["conclusion"] = essay2parainfo[essay_id]["conclusion"]
        para_info_dict[para_id]["context"] = essay2paraids[essay_id]

    return essay_info_dict, essay_max_n_dict, para_info_dict


def get_essay_detail(essay_lines, max_n_spans):
    essay_id = int(essay_lines[0].split("\t")[0])
    # list of (start_idx, end_idx) of each ac span
    ac_spans = []
    # text of each span
    ac_texts = []
    # type of each span (premise, claim, majorclaim)
    ac_types = []
    # in which type of paragraph each ac is (opening, body, ending)
    ac_paratypes = []
    # id of the paragraph where the ac appears
    ac_paras = []
    # id of each ac (in paragraoh)
    ac_positions_in_para = []
    # linked acs (source_ac, target_ac, relation_type)
    ac_relations = []
    # list of (startr_idx, end_idx) of each am span
    shell_spans = []
    relation2id = {"Support": 0, "Attack": 1}
    actype2id = {"Premise": 0, "Claim": 1, "Claim:For": 1, "Claim:Against": 1, "MajorClaim": 2}
    paratype2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}

    relation_type_seq = np.zeros(max_n_spans).astype('int32')
    relation_type_seq.fill(2)

    text = [line.strip().split("\t")[-1].lower()
        for line in essay_lines]

    previous_span_end = 0
    for ac_type, lines in itertools.groupby(essay_lines, key=lambda x: x.split("\t")[6]):
        ac_lines = list(lines)

        if ac_lines[0].split("\t")[7] != "-":
            ac_text = [ac_line.split("\t")[-1].strip() for ac_line in ac_lines]
            ac_texts.append(ac_text)

            para_i = int(ac_lines[0].split("\t")[2])
            para_type = ac_lines[0].split("\t")[3]
            ac_i = int(ac_lines[0].split("\t")[6])
            ac_type = ac_lines[0].split("\t")[7]
            start = int(ac_lines[0].split("\t")[11])
            end = int(ac_lines[-1].split("\t")[11])

            ac_positions_in_para.append(ac_i)
            ac_types.append(actype2id[ac_type])
            ac_paratypes.append(para_type)
            ac_paras.append(para_i)

            ac_span = (start, end)
            ac_spans.append(ac_span)

            shell_span = get_shell_lang_span(start, text, previous_span_end)
            shell_spans.append(shell_span)

            if ac_type == "Claim:For":
                relation_type_seq[ac_i] = 0
            elif ac_type == "Claim:Against":
                relation_type_seq[ac_i] = 1

            if "Claim" not in ac_lines[0].split("\t")[7]:
                ac_relations.append(
                   (ac_i,
                    ac_i + int(ac_lines[0].split("\t")[8]),
                    relation2id[ac_lines[0].split("\t")[9].strip()]))
                relation_type_seq[ac_i] = relation2id[ac_lines[0].split("\t")[9].strip()]
            previous_span_end = end

    assert len(ac_spans) == len(ac_positions_in_para)
    assert len(ac_spans) == len(ac_types)
    assert len(ac_spans) == len(ac_paratypes)
    assert len(ac_spans) == len(ac_paras)
    assert len(ac_spans) == len(shell_spans)
    assert len(relation_type_seq) == max_n_spans

    assert max(relation_type_seq).tolist() <= 2
    assert len(ac_spans) >= len(ac_relations)

    n_acs = len(ac_spans)
    relation_type_seq[n_acs:] = -1

    relation_matrix = relation_info2relation_matrix(ac_relations,
                                                    max_n_spans,
                                                    n_acs)

    assert len(relation_matrix) == max_n_spans*max_n_spans

    relation_targets, _, relation_depth = \
        relation_info2target_sequence(ac_relations,
                                      ac_types,
                                      max_n_spans,
                                      n_acs)

    assert len(relation_targets) == max_n_spans
    assert len(relation_depth) == max_n_spans

    relation_children = relation_info2children_sequence(ac_relations,
                                                        max_n_spans,
                                                        n_acs)
    # what's the meaning?
    # [0:12]: forward position
    # [12:24]: backward position
    # [24:28]: paragraph type
    ac_position_info = np.array([
                       ac_positions_in_para,
                       [(i_ac - max(ac_positions_in_para))*(-1)+max_n_spans
                        for i_ac in ac_positions_in_para],
                       [paratype2id[i]+2*max_n_spans for i in ac_paratypes]], dtype=np.int32).T
    print(ac_position_info)
    print(ac_paratypes)
    print(ac_positions_in_para)
    print("****")

    assert ac_position_info.shape == (n_acs, 3)

    if not len(ac_position_info):
        para_type = ac_lines[0].split("\t")[3]
        ac_position_info = np.array([[0,
                                      0 + max_n_spans, paratype2id[para_type] + max_n_spans*2]], 
                                    dtype=np.int32)

    essay_detail_dict = {}
    essay_detail_dict["essay_id"] = essay_id
    essay_detail_dict["text"] = text
    essay_detail_dict["ac_spans"] = ac_spans
    essay_detail_dict["shell_spans"] = shell_spans
    essay_detail_dict["ac_types"] = np.pad(ac_types,
                                           [0, max_n_spans-len(ac_types)],
                                           'constant',
                                           constant_values=(-1, -1))
    essay_detail_dict["ac_paratypes"] = ac_paratypes
    essay_detail_dict["ac_paras"] = ac_paras
    essay_detail_dict["ac_position_info"] = ac_position_info
    essay_detail_dict["relation_matrix"] = relation_matrix
    essay_detail_dict["relation_targets"] = relation_targets
    essay_detail_dict["relation_children"] = relation_children
    essay_detail_dict["ac_relation_types"] = relation_type_seq
    essay_detail_dict["ac_relation_depth"] = relation_depth

    return essay_detail_dict


def relation_info2target_sequence(ac_relations, ac_types, max_n_spans, n_spans):

    """Summary line.

    Args:
        arg1 (list): list of argumentative information tuple (source, target, relation_type)

    Returns:
        array: array of target ac index
    """

    relation_seq = np.zeros(max_n_spans).astype('int32')
    relation_type_seq = np.zeros(max_n_spans).astype('int32')
    direction_seq = np.zeros(max_n_spans).astype('int32')
    depth_seq = np.zeros(max_n_spans).astype('int32')

    relation_seq.fill(max_n_spans)
    relation_type_seq.fill(2)
    direction_seq.fill(2)
    depth_seq.fill(100)
    relation_seq[n_spans:] = -1
    relation_type_seq[n_spans:] = -1
    direction_seq[n_spans:] = -1
    depth_seq[n_spans:] = -1

    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        relation_seq[source_i] = target_i
        relation_type_seq[source_i] = combination[2]

    for i in range(len(relation_seq)):
        depth = 0
        target_i = relation_seq[int(i)]
        if target_i == -1:
            continue
        while(1):
            if target_i == max_n_spans:
                break
            else:
                target_i = relation_seq[int(target_i)]
                depth += 1
        depth_seq[i] = depth

    return relation_seq, relation_type_seq, depth_seq


def relation_info2children_sequence(ac_relations, max_n_spans, n_spans):
    children_list = [[] for _ in range(max_n_spans + 1)]

    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        children_list[target_i].append(source_i)
    return children_list


def relation_info2relation_matrix(ac_relations, max_n_spans, n_spans):

    """Summary line.

    Args:
        arg1 (list): list of argumentative information tuple (source, target, relation_type)

    Returns:
        ndarray: flatten version of relation matrix
                                 (axis0: source_AC_id,
                                  axis1: target_AC_id,
                                  value: relation type)
    """

    relation_matrix = np.zeros((max_n_spans, max_n_spans)).astype('int32')
    relation_matrix.fill(-1)
    relation_matrix[n_spans:, :] = -1
    relation_matrix[:, n_spans:] = -1
    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        relation_type = combination[2]
        relation_matrix[source_i, target_i] = relation_type

    return relation_matrix.flatten().astype('int32')

def get_shell_lang_span(start, text, previous_span_end):

    EOS_tokens_list = [".",
                       "!",
                       "?",
                       "</AC>",
                       "</para-intro>",
                       "</para-body>",
                       "</para-conclusion>",
                       "</essay>"]

    # EOS_ids_set = set([vocab[token.lower()
    #                    for token in EOS_tokens_list if token.lower() in vocab])
    EOS_ids_set = set([token.lower()
                       for token in EOS_tokens_list])
    shell_lang = []
    if start == 0:
        shell_span = (start, start)
        return shell_span

    for i in range(start-1, previous_span_end, -1):
        if text[int(i)] not in EOS_ids_set:
            shell_lang.append(int(i))
        else:
            break
    if shell_lang:
        shell_start = min(shell_lang)
        shell_end = max(shell_lang)
        shell_span = (shell_start, shell_end)
    else:
        shell_span = (start-1, start-1)
    return shell_span


def load_data(essay_ids, essay_info_dict, max_n_spans, args):

    ts_link = np.array([essay_info_dict[int(i)]["relation_targets"]
                        for i in list(essay_ids)], dtype=np.int32)
    ts_type = np.array([essay_info_dict[int(i)]["ac_types"]
                        for i in list(essay_ids)], dtype=np.int32)
    ts_link_type = np.array([essay_info_dict[int(i)]["ac_relation_types"]
                             for i in list(essay_ids)], dtype=np.int32)

    return list(zip(essay_ids,
                    ts_link,
                    ts_type,
                    ts_link_type))


def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip(r"/")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        print ('the directory already existes!')
        return False

    
def DataLoader(data_file, split_test_file):
    # /home/baojianzhu/workspace/trans_am_test/Data/pe_data_df.csv
    # /home/baojianzhu/workspace/trans_am_test/Data/test_paragraph_index.json
    data_df = pd.read_csv(data_file)
    with open(split_test_file, "r") as fp:
        test_id_list = json.load(fp)
    test_data_df = data_df[data_df["essay_id"].isin(test_id_list)]
    train_data_df = data_df[~data_df["essay_id"].isin(test_id_list)]

    # # split with essay level
    # essay_id2parag_id_dict = train_data_df.groupby("essay_id").groups
    # essay_id_list = list(essay_id2parag_id_dict.keys())
    # random.shuffle(essay_id_list)
    # num_train_essay = int(len(essay_id_list) * 0.9)
    # train_essay_id_list = essay_id_list[:num_train_essay]
    # dev_essay_id_list = essay_id_list[num_train_essay:]

    # train_para_id_list = []
    # for essay_id in train_essay_id_list:
    #     train_para_id_list += essay_id2parag_id_dict[essay_id].tolist()
    
    # dev_para_id_list = []
    # for essay_id in dev_essay_id_list:
    #     dev_para_id_list += essay_id2parag_id_dict[essay_id].tolist()
    
    num_train_para = int(len(train_data_df) * 0.8)
    para_id_list = list(train_data_df["para_id"])
    random.shuffle(para_id_list)
    dev_para_id_list = para_id_list[num_train_para:]
    dev_data_df = train_data_df[train_data_df["para_id"].isin(dev_para_id_list)]
    train_data_df = train_data_df[~train_data_df["para_id"].isin(dev_para_id_list)]
    return train_data_df, dev_data_df, test_data_df
    # return train_data_df, test_data_df

def add_column_to_df(data_df, choosen_add_list, column_name):
    choosen_index_list = list(data_df["adu_spans"].apply(lambda x: len(eval(x))>1))
    choosen_idx = 0
    all_add_list = []
    for i in range(len(data_df)):
        if choosen_index_list[i]:
            all_add_list.append(choosen_add_list[choosen_idx])
            choosen_idx += 1
        else:
            all_add_list.append(None)
    data_df[column_name] = all_add_list
    return data_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, \
                        default='pe_token_level_data.tsv')
    parser.add_argument("--data-save-path", type=str, \
                        default='pe_data_df.csv')
    parser.add_argument("--vocab-save-path", type=str, \
                        default='bow_vocab.json')
    args = parser.parse_args()
    
    essay_info_dict_list, essay_max_n_dict, para_info_dict = get_data_dicts(args.data_path)
    df_dict = { "essay_id": [],
                "para_id": [],
                "para_types": [],
                "para_text": [], 
                "adu_spans": [], 
                "ac_spans": [], 
                "ai_spans": [],
                "ac_types": [],
                "ac_rel_targets": [],
                "ac_rel_types": [], 
                "ac_rel_pairs": []}
    punc_set = set([',', '?', '!', '.', '-', ';'])
    id2rel = {0: "Support", 1: "Attack", 2: "MC_rel"}
    id2ac_type = {0: "Premise", 1: "Claim", 2: "MajorClaim"}
    id2para_type = {0: "intro", 1: "body", 2: "conclusion", 3: "prompt"}
    text2para_type = {"<prompt>": "prompt",
                      "<para-intro>": "intro",
                      "<para-body>": "body",
                      "<para-conclusion>": "conclusion"}
    
    for global_para_id, para_info_dict in essay_info_dict_list.items():
        df_dict["para_text"].append(' '.join(para_info_dict["text"]))
        ac_spans = para_info_dict["ac_spans"]
        ai_spans = para_info_dict["shell_spans"]
        df_dict["ac_spans"].append(ac_spans)
        df_dict["ai_spans"].append(ai_spans)
        # df_dict["para_types"].append("prompt" if len(para_info_dict["ac_paratypes"]) == 0 else para_info_dict["ac_paratypes"][0])
        df_dict["para_types"].append(text2para_type[para_info_dict["text"][0]])
        
        adu_spans = []
        for ac_span, ai_span in zip(ac_spans, ai_spans):
            # if ai_span[0] == ai_span[1] \
            # and para_info_dict["text"][ai_span[0]] in punc_set:
            #     adu_span = ac_span
            # else:
            #     adu_span = (ai_span[0], ac_span[1])
            adu_span = (ai_span[0], ac_span[1])
            adu_spans.append(adu_span)
        df_dict["adu_spans"].append(adu_spans)
        ac_types = para_info_dict["ac_types"][:len(ac_spans)].tolist()
        ac_rel_targets = para_info_dict["relation_targets"][:len(ac_spans)].tolist()
        ac_rel_types = para_info_dict["ac_relation_types"][:len(ac_spans)].tolist()
        ac_types = [id2ac_type[type_id] for type_id in ac_types]
        ac_rel_targets = [target_id for target_id in ac_rel_targets]
        ac_rel_types = [id2rel[rel_id] for rel_id in ac_rel_types]
        df_dict["ac_types"].append(ac_types)
        df_dict["ac_rel_targets"].append(ac_rel_targets)
        # df_dict["ac_rel_types"].append(ac_rel_types)
        ac_rel_pairs = []
        ac_rel_types_noclaim = []
        for child, parent in enumerate(ac_rel_targets):
            if parent != 12:
                ac_rel_pairs.append((parent, child))
                ac_rel_types_noclaim.append(ac_rel_types[child])
        df_dict["ac_rel_types"].append(ac_rel_types_noclaim)
        df_dict["ac_rel_pairs"].append(ac_rel_pairs)
        df_dict["para_id"].append(global_para_id)
        df_dict["essay_id"].append(para_info_dict["essay_id"])

    pe_data_df = pd.DataFrame(df_dict)
    SCRIPT_PATH = os.path.dirname(__file__)
    DATA_SAVE_PATH = os.path.join(SCRIPT_PATH, args.data_save_path)
    pe_data_df.to_csv(DATA_SAVE_PATH, index=False)


        