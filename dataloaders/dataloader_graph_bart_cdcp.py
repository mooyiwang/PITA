from torch.utils.data import Dataset
import pandas as pd
import json
import torch
from transformers import AutoTokenizer
import sys
import numpy as np
from utils.basic_utils import load_json
from configs.config import shared_configs

class ArgMiningDataset(Dataset):
    
    def __init__(self, data_df, config):
        ac_type2id = config.ac_type2id
        para_type2id = config.para_type2id
        rel_type2id = config.rel_type2id

        para_text_list = list(data_df['para_text'])
        orig_AC_spans_list = [eval(AC_spans) for AC_spans in data_df['adu_spans']]
        # AC_bow_token_ids_lists = []
        # for para_text, AC_spans in zip(para_text_list, orig_AC_spans_list):
        #     para_tokens = para_text.split(' ')
        #     para_token_ids = [vocab_dict[token] if token in vocab_dict \
        #                                         else vocab_dict['<unk>'] \
        #                                         for token in para_tokens]
        #     AC_token_ids_list = []
        #     for AC_span in AC_spans:
        #         AC_token_ids = para_token_ids[AC_span[0]:AC_span[1]+1]
        #         AC_token_ids_list.append(AC_token_ids)
        #     AC_bow_token_ids_lists.append(AC_token_ids_list)

        # bow_lists = []
        # eye = torch.tensor(np.identity(len(self.vocab_dict), dtype=np.float32))
        # for AC_bow_token_ids_list in AC_bow_token_ids_lists:
        #     bow_list = []
        #     for AC_bow_token_ids in AC_bow_token_ids_list:
        #         bow_list.append(torch.sum(eye[AC_bow_token_ids], dim=0))
        #     bow_lists.append(torch.stack(bow_list))
        # self.AC_bow_array_list = bow_lists

        self.special_tokens = ['<essay>', '<para-conclusion>', '<para-body>', '<para-intro>', '<ac>',
                               '</essay>', '</para-conclusion>', '</para-body>', '</para-intro>', '</ac>']
        self.special_tokens_dict = {'additional_special_tokens': self.special_tokens}
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_weights_path)
        self.tokenizer.add_special_tokens(self.special_tokens_dict)  ############为 roberta 设置特殊字符

        AC_spans_list = []
        para_token_ids_list = []
        max_sample_len = 0
        for para_text, AC_spans in zip(para_text_list, orig_AC_spans_list):
            orig_pos2bert_pos = {}
            para_tokens = para_text.split(' ')
            para_tokens_for_bert = []
            para_tokens_for_bert = ['<s>']
            for orig_pos, token in enumerate(para_tokens):
                if token not in self.special_tokens:
                    bert_tokens = self.tokenizer.tokenize(token)
                else:
                    bert_tokens = [token]
                cur_len = len(para_tokens_for_bert)
                orig_pos2bert_pos[orig_pos] = (cur_len, cur_len+len(bert_tokens)-1)
                para_tokens_for_bert += bert_tokens
            para_tokens_for_bert.append('</s>')
            para_token_ids_list.append(self.tokenizer.convert_tokens_to_ids(para_tokens_for_bert)) # word to id in bert

            if max_sample_len < len(para_token_ids_list[-1]):
                max_sample_len = len(para_token_ids_list[-1])

            AC_spans_for_bert = []
            for AC_span in AC_spans:
                start = orig_pos2bert_pos[AC_span[0]][0]
                end = orig_pos2bert_pos[AC_span[1]][1]
                AC_spans_for_bert.append((start, end))
            AC_spans_list.append(AC_spans_for_bert)
        self.AC_spans_list = AC_spans_list
        self.para_token_ids_list = para_token_ids_list

        # para_types_list = list(data_df["para_types"])
        self.AC_types_list = [list(map(lambda x: ac_type2id[x], eval(AC_types))) \
                                for AC_types in data_df["ac_types"]]
        print("max_sample_len", max_sample_len)
        # AC_positions_list = []
        # AC_para_types_list = []
        #
        # for AC_spans, para_type in zip(self.AC_spans_list, para_types_list):
        #     AC_positions = list(range(len(AC_spans)))
        #     AC_positions_list.append(AC_positions)
        #     AC_para_types_list.append([para_type2id[para_type]] * len(AC_spans))
        # self.AC_positions_list = AC_positions_list
        # self.AC_para_types_list = AC_para_types_list

        self.AR_pairs_list = [eval(_) for _ in data_df['ac_rel_pairs']] # ac_types,ac_rel_targets,ac_rel_types,ac_rel_pairs
        self.AR_link_types_list = [list(map(lambda x: rel_type2id[x], eval(AC_link_types))) \
                              for AC_link_types in data_df["ac_rel_types"]]

        self.whole_graph_list = [eval(_) for _ in data_df['whole_graph']]

        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        one_sample = (
            self.para_token_ids_list[index],
            # self.AC_bow_array_list[index],
            self.AC_spans_list[index],
            self.whole_graph_list[index],
            # self.AC_positions_list[index],
            # self.AC_para_types_list[index],
            # self.parser_states_list[index],
            self.AC_types_list[index],
            self.AR_pairs_list[index],
            self.AR_link_types_list[index]
        )
        return one_sample

def generate_batch_fn(batch):
    batch = list(zip(*batch))
    batch = {
        'para_tokens_ids': batch[0],
        # 'bow_vecs': batch[1],
        'AC_spans': batch[1],
        'whole_graph': batch[2],
        # 'AC_positions': batch[3],
        # 'AC_para_types': batch[4],
        'true_AC_types': batch[3],
        'true_AR_pairs': batch[4], # (parent, child)
        'true_AR_link_types': batch[5],
    }
    return batch


class InfiniteIterator(object):
    """iterate an iterable oobject infinitely"""
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(iterable)

    def __iter__(self):
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.iterable)
                batch = next(self.iterator)
            yield batch
            


if __name__ == '__main__':
    config = shared_configs.get_pe_args()
    
    data_path = './data/cdcp/cdcp_data_df_graphs6.csv'
    data_df = pd.read_csv(data_path)
    data_df = data_df[data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]

    # vocab_dict = load_json(config.vocab_path)
    train_dataset = ArgMiningDataset(data_df, config)
    print(train_dataset)