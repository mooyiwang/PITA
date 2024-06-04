import os
import torch
import sys
# from pytorch_pretrained_bert import BertTokenizer
from transformers import BertTokenizer
import pandas as pd

bert_path = "../bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

if __name__ == "__main__":

    orig_data_path = "cdcp_data_df_noabbre_srl.csv"
    orig_data_df = pd.read_csv(orig_data_path)
    orig_text_list = list(orig_data_df["para_text_replace_abbre"])
    orig_AC_spans_list = [eval(AC_spans) for AC_spans in orig_data_df['adu_spans_noabbre']]

    orig_para_srl_list = []
    for para_srl in orig_data_df['para_srl']:
	    # print(para_srl)
	    # print(eval(para_srl))
	    orig_para_srl_list.append(eval(para_srl))
    
    tokenized_text_list = []
    new_adu_spans_list = []
    new_para_srl_list = []

    special_tokens = {'<pad>', '<essay>', '<para-conclusion>',
                           '<para-body>', '<para-intro>', '<ac>',
                           '</essay>', '</para-conclusion>', '</para-body>',
                           '</para-intro>', '</ac>'}
    for para_text, AC_spans, para_srl_list in zip(orig_text_list, orig_AC_spans_list, orig_para_srl_list):
        orig_pos2bert_pos = {}
        para_tokens = para_text.split(' ')
        tokenized_text_tokens = []
        for orig_pos, token in enumerate(para_tokens):
            if token not in special_tokens:
                bert_tokens = bert_tokenizer.tokenize(token)
            else:
                bert_tokens = [token]
            cur_len = len(tokenized_text_tokens)
            orig_pos2bert_pos[orig_pos] = (cur_len, cur_len + len(bert_tokens) - 1)
            tokenized_text_tokens += bert_tokens
        tokenized_text_list.append(' '.join(tokenized_text_tokens))
    
        AC_spans_for_bert = []
        for AC_span in AC_spans:
            start = orig_pos2bert_pos[AC_span[0]][0]
            end = orig_pos2bert_pos[AC_span[1]][1]
            AC_spans_for_bert.append((start, end))
        new_adu_spans_list.append(AC_spans_for_bert)
    
        para_srl_for_bert = []
        for sentence_srl_list, AC_span in zip(para_srl_list, AC_spans):
            sentence_srl_for_bert = []
            # print("AC_span, sentence_srl_list ", AC_span, sentence_srl_list)
            # print('orig_pos2bert_pos ', orig_pos2bert_pos)
            for tuple_srl_list in sentence_srl_list:
                # if len(tuple_srl_list) == 0:
                #     print("para_srl_list, AC_spans ", para_srl_list, AC_spans)
                # print(tuple_srl_list)
                tuple_srl_for_bert = []
                for tuple in tuple_srl_list:
                    start = orig_pos2bert_pos[tuple[0] + AC_span[0]][0] - orig_pos2bert_pos[AC_span[0]][0]
                    end = orig_pos2bert_pos[tuple[1] + AC_span[0]][1] - orig_pos2bert_pos[AC_span[0]][0]
                    tuple_srl_for_bert.append((start, end, tuple[2]))
            
                sentence_srl_for_bert.append(tuple_srl_for_bert)
            para_srl_for_bert.append(sentence_srl_for_bert)
        new_para_srl_list.append(para_srl_for_bert)
    
    # for orig_text in orig_text_list:
    #     tokenized_text_tokens = []
    #     adu_spans = []
    #     start = 0
    #     end = 0
    #     token_id = 0
    #     for word in orig_text.split(' '):
    #         if word == '<para-body>' or word == '</para-body>':
    #             pass
    #             token_id += 1
    #             tokenized_text_tokens.append(word)
    #         elif word == '<ac>':
    #             start = token_id
    #             token_id += 1
    #             tokenized_text_tokens.append(word)
    #         elif word == '</ac>':
    #             end = token_id
    #             adu_spans.append((start, end))
    #             token_id += 1
    #             tokenized_text_tokens.append(word)
    #         else:
    #             tokens = bert_tokenizer.tokenize(word)
    #             token_id += len(tokens)
    #             tokenized_text_tokens.extend(tokens)
    #     tokenized_text_list.append(' '.join(tokenized_text_tokens))
    #     new_adu_spans_list.append(adu_spans)
    
    orig_data_df["para_tokenized_text"] = tokenized_text_list
    orig_data_df["tokenized_adu_spans"] = new_adu_spans_list
    orig_data_df['tokenized_para_srl'] = new_para_srl_list
    
    
    truncated_adu_spans_list = list(orig_data_df['tokenized_adu_spans'])
    truncated_para_srl_list = list(orig_data_df['tokenized_para_srl'])
    truncated_para_srl_role_matrix_list = list(map(eval, orig_data_df['para_srl_role_matrix']))
    truncated_ac_types_list = list(map(eval, orig_data_df['ac_types']))
    truncated_ac_rel_pairs_list = list(map(eval, orig_data_df['ac_rel_pairs']))
    truncated_ac_rel_types_list = list(map(eval, orig_data_df['ac_rel_types']))

    droped_ac = 0
    droped_rel = 0

    for idx, (para_text, adu_spans, ac_types, ac_rel_pairs, ac_rel_types, para_srl, para_srl_role_matrix) \
        in enumerate(zip(orig_data_df['para_tokenized_text'], truncated_adu_spans_list,
                truncated_ac_types_list, truncated_ac_rel_pairs_list, 
                truncated_ac_rel_types_list, truncated_para_srl_list, truncated_para_srl_role_matrix_list)):
        # print(para_text, adu_spans, ac_types, ac_rel_pairs, ac_rel_types, para_srl, para_srl_role_matrix)
        # break
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
            new_para_srl = para_srl[:exceeded_idx]
            new_para_srl_role_matrix = para_srl_role_matrix[:exceeded_idx]
            
            new_ac_rel_pairs = []
            new_ac_rel_types = []
            for pairs, pair_type in zip(ac_rel_pairs, ac_rel_types):
                if pairs[0] < exceeded_idx and pairs[1] < exceeded_idx:
                    new_ac_rel_pairs.append(pairs)
                    new_ac_rel_types.append(pair_type)
                else:
                    droped_rel += 1
            truncated_adu_spans_list[idx] = new_adu_spans
            truncated_para_srl_list[idx] = new_para_srl
            truncated_para_srl_role_matrix_list[idx] = new_para_srl_role_matrix
            truncated_ac_types_list[idx] = new_ac_types
            truncated_ac_rel_pairs_list[idx] = new_ac_rel_pairs
            truncated_ac_rel_types_list[idx] = new_ac_rel_types
    
    orig_data_df['trunc_adu_spans'] = truncated_adu_spans_list
    orig_data_df['trunc_para_srl'] = truncated_para_srl_list
    orig_data_df['trunc_para_srl_role_matrix'] = truncated_para_srl_role_matrix_list
    orig_data_df['trunc_ac_types'] = truncated_ac_types_list
    orig_data_df['trunc_ac_rel_pairs'] = truncated_ac_rel_pairs_list
    orig_data_df['trunc_ac_rel_types'] = truncated_ac_rel_types_list

    orig_data_df.to_csv("cdcp_data_bert_df_noabbre_srl.csv", index=False)
    # with open('CDCP4bert_srl.tsv', 'w') as fp:
    #     for text in orig_data_df['para_text']:
    #         fp.write(text + '\n')
    #
    # print('droped ac: {}'.format(droped_ac))
    # print('droped rel: {}'.format(droped_rel))




