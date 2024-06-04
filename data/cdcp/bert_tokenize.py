import os
import torch
import sys
from pytorch_pretrained_bert import BertTokenizer
import pandas as pd

bert_path = "data/bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

if __name__ == "__main__":

    orig_data_path = "data/cdcp/cdcp_data_df_srl2.csv"
    orig_data_df = pd.read_csv(orig_data_path)
    orig_text_list = list(orig_data_df["para_text"])
    tokenized_text_list = []
    new_adu_spans_list = []
    for orig_text in orig_text_list:
        tokenized_text_tokens = []
        adu_spans = []
        start = 0
        end = 0
        token_id = 0
        for word in orig_text.split(' '):
            if word == '<para-body>' or word == '</para-body>':
                pass
                token_id += 1
                tokenized_text_tokens.append(word)
            elif word == '<ac>':
                start = token_id
                token_id += 1
                tokenized_text_tokens.append(word)
            elif word == '</ac>':
                end = token_id
                adu_spans.append((start, end))
                token_id += 1
                tokenized_text_tokens.append(word)
            else:
                tokens = bert_tokenizer.tokenize(word)
                token_id += len(tokens)
                tokenized_text_tokens.extend(tokens)
        tokenized_text_list.append(' '.join(tokenized_text_tokens))
        new_adu_spans_list.append(adu_spans)
    orig_data_df["para_text"] = tokenized_text_list
    orig_data_df["adu_spans"] = new_adu_spans_list
    
    truncated_adu_spans_list = list(orig_data_df['adu_spans'])
    truncated_ac_types_list = list(map(eval, orig_data_df['ac_types']))
    truncated_ac_rel_pairs_list = list(map(eval, orig_data_df['ac_rel_pairs']))
    truncated_ac_rel_types_list = list(map(eval, orig_data_df['ac_rel_types']))

    droped_ac = 0
    droped_rel = 0

    for idx, (para_text, adu_spans, ac_types, ac_rel_pairs, ac_rel_types) \
        in enumerate(zip( orig_data_df['para_text'], truncated_adu_spans_list, 
                truncated_ac_types_list, truncated_ac_rel_pairs_list, 
                truncated_ac_rel_types_list)):
        if len(para_text.split(' ')) > 510:
            exceeded_idx = len(adu_spans)
            for i, span in enumerate(adu_spans):
                if span[0] > 510 or span[1] > 510:
                    exceeded_idx = i
                    break
            if orig_data_df['dataset_type'][idx] == 'test':
                droped_ac += len(adu_spans) - exceeded_idx
            new_adu_spans = adu_spans[:exceeded_idx]
            new_ac_types = ac_types[:exceeded_idx]
            new_ac_rel_pairs = []
            new_ac_rel_types = []
            for pairs, pair_type in zip(ac_rel_pairs, ac_rel_types):
                if pairs[0] < exceeded_idx and pairs[1] < exceeded_idx:
                    new_ac_rel_pairs.append(pairs)
                    new_ac_rel_types.append(pair_type)
                else:
                    droped_rel += 1
            truncated_adu_spans_list[idx] = new_adu_spans
            truncated_ac_types_list[idx] = new_ac_types
            truncated_ac_rel_pairs_list[idx] = new_ac_rel_pairs
            truncated_ac_rel_types_list[idx] = new_ac_rel_types
    orig_data_df['trunc_adu_spans'] = truncated_adu_spans_list
    orig_data_df['trunc_ac_types'] = truncated_ac_types_list
    orig_data_df['trunc_ac_rel_pairs'] = truncated_ac_rel_pairs_list
    orig_data_df['trunc_ac_rel_types'] = truncated_ac_rel_types_list

    orig_data_df.to_csv("data/cdcp/cdcp_data_bert_df.csv", index=False)
    with open('data/cdcp/CDCP4bert.tsv', 'w') as fp:
        for text in orig_data_df['para_text']:
            fp.write(text + '\n')
    
    print('droped ac: {}'.format(droped_ac))
    print('droped rel: {}'.format(droped_rel))




