from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import nltk
from pathlib import Path
import os, time
import pandas as pd
import numpy as np

#
# sentence_role_list =  [[(6, 6, 'V'), (7, 7, 'ARG1')],
#                        [(1, 3, 'ARGM-MNR'), (5, 7, 'ARG0'), (9, 9, 'V'), (10, 23, 'ARG1')],
#                        [(13, 13, 'V'), (14, 23, 'ARG1')],
#                        [(14, 14, 'ARG0'), (16, 16, 'R-ARG0'), (17, 17, 'V'), (18, 23, 'ARG1')],
#                        [(14, 14, 'ARG3'), (16, 16, 'R-ARG3'), (17, 17, 'ARGM-MOD'), (18, 18, 'V'), (19, 23, 'ARG2')]]


def get_interaction_matrix(s_role_list):
    flat_role_list = [tuple for tuple_role_list in s_role_list for tuple in tuple_role_list]
    node_num = len(flat_role_list)
    
    node_link = np.zeros([node_num, node_num])
    for i in range(node_num):
        node_link[i][i] = 1
        for j in range(i, node_num):
            if flat_role_list[i][0] <= flat_role_list[j][1] and flat_role_list[i][1] >= flat_role_list[j][
                0]:  # max(flat_role_list[i][0], flat_role_list[i][1]) <= min(flat_role_list[j][0], flat_role_list[j][1])
                node_link[i][j] = 1
                node_link[j][i] = 1
    
    accumulated = 0
    for k, tuple_role_list in enumerate(s_role_list):
        sub_node_num = len(tuple_role_list)
        sub_node_link = np.ones([sub_node_num, sub_node_num])
        node_link[accumulated:sub_node_num + accumulated, accumulated:sub_node_num + accumulated] = sub_node_link
        accumulated += sub_node_num
    
    return node_link


# nltk.download('punkt')
# nltk.download('wordnet')

# CACHE_ROOT = Path(os.getenv("ALLENNLP_CACHE_ROOT", Path.home() / ".allennlp"))
# CACHE_DIRECTORY = str(CACHE_ROOT / "cache")
#
# print(CACHE_DIRECTORY)

start = time.time()

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", cuda_device=0)

data_path= "./data/cdcp/cdcp_data_df_noabbre.csv"
data_df = pd.read_csv(data_path)

# data_df = data_df[data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]
para_text_list = list(data_df['para_text_replace_abbre'])
orig_AC_spans_list = [eval(AC_spans) for AC_spans in data_df['adu_spans_noabbre']]

special_tokens = {'<pad>', '<essay>', '<para-conclusion>',
                       '<para-body>', '<para-intro>', '<ac>',
                       '</essay>', '</para-conclusion>', '</para-body>',
                       '</para-intro>', '</ac>'}

AC_spans_list = []
tags_list = []

role_chars = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'R-ARG0','R-ARG1','R-ARG2','R-ARG3','R-ARG4','C-ARG0','C-ARG1','C-ARG2','C-ARG3', 'C-ARG4',
              'V','DIS', 'ADJ', 'TMP', 'CAU', 'MNR', 'EXT', 'ADV', 'LOC', 'PRP', 'NEG', 'PRD', 'COM',
              'GOL', 'DIR', 'PNC']
# R-ARG0,1,2,3,4
# C-ARG0,1,2,3,4
# V
# DIS
# ADJ
# TMP
# CAU
# MNR
# EXT
# ADV
# CAU
# LOC
# PRP
# NEG
# PRD
# COM
# GOL
# DIR
# PNC

# REC 待定
# LVB 待定

EOS_tokens_list = [".", "!", "?", "_",  '<ac>', '</ac>']

# illeage_char  = {"id": "identity", "cannot": "not", "dont":"not", "doesnt": "not", "wont":"not", "cant":"not",
#                  "youre":"you", "shes", "youre", "1pm","1pm", "1pm","1pm","1pm","1pm","1pm","1pm","1pm","1pm","1pm"}
map = {'cannot': 'can not', 'id': 'identity', 'dont' : "do not", 'wont': 'will not', 'cant': 'can not',
       'youre': 'you are', 'arent': 'are not', 'shes': 'she is', '10pm': '10 pm', '9pm': '9 pm', '9am': '9 am',
        '8pm': '8 pm', '8am': '8 am',
       '7pm': '7 pm', '10M': '10 M', '1692g': '1692 g', '10am': '10 am', '5pm': '5 pm', '12K': '12 K',
       'Thats': 'That is', 'thats': 'that is', 'hes': 'he is', 'doesnt': 'does not', 'im': 'i m'}

para_srl_list = []
para_srl_role_matrix_list = []
for para_text, AC_spans in zip(para_text_list, orig_AC_spans_list):
    if len(AC_spans) == 0:
        para_srl_list.append([])
        para_srl_role_matrix_list.append([])
        continue
        
    orig_pos2bert_pos = {}
    para_tokens = para_text.split(' ')
    # print(para_tokens)
    sentences_srl_list = []
    sentences_srl_role_matrix_list = []
    for index, spans in enumerate(AC_spans):
        sentence = para_tokens[spans[0]: spans[1]+1]
        # print(sentence)
        ac_idx = sentence.index('<ac>')
        ace_idx = sentence.index("</ac>")
        
        assert ac_idx == 0
        assert ace_idx == len(sentence) - 1
        
        sentence.remove('<ac>')
        sentence.remove('</ac>')
        # print(sentence)
        
        # print(sentence)
        tuples = predictor.predict(sentence=" ".join(sentence))
        sentence_role_list = []
        # print(tuples)
        
        if len(tuples['words']) != len(sentence):
            gap_num = len(tuples['words']) - len(sentence)
            illegal_char = []
            for i in range(sentence):
                if sentence[i] in map.keys():
                    illegal_char.append([i, sentence[i]])

            assert len(illegal_char) == gap_num
            
            
            
        
        for tuple in tuples['verbs']:
            tags = tuple['tags']
            if 'B-ARG' in "".join(tags):
                # print(tags)
                tags_list.extend(tags)
                
                if len(sentence) != len(tags):
                    print("length not equal \n", sentence, "\n", tags, "\n", " ".join(sentence))
                    

                tuple_role_list = []
                pre_ind = 0
                pre_role = ''

                tags.insert(ac_idx, 'O')
                tags.insert(ace_idx, 'O')
                
                # print(tags)
                
                for idx, tag in enumerate(tags):
                    if pre_role != '' and pre_role != tag[2:]:
                        if ("ARG" in pre_role or "V" in pre_role) and pre_role.split("-")[-1] in role_chars:
                            tuple_role_list.append((pre_ind, idx-1, pre_role))
                        pre_ind = idx
                        pre_role = tag[2:]
                    
                    if tag == 'O':
                        pre_ind = idx + 1
                        pre_role = ''
                        continue

                    if pre_role == '' and pre_role != tag[2:]:
                        pre_role = tag[2:]

                    elif pre_role != '' and pre_role == tag[2:]:
                        if idx == len(tags)-1:
                            if ("ARG" in pre_role or "V" in pre_role) and pre_role.split("-")[-1] in role_chars:
                                tuple_role_list.append((pre_ind, idx, pre_role))
                        
                # print(tuple_role_list)
                if len(tuple_role_list) > 0:
                    sentence_role_list.append(tuple_role_list)

        if len(sentence_role_list) != 0:
            sentence_role_matrix = get_interaction_matrix(sentence_role_list)
            sentences_srl_role_matrix_list.append(sentence_role_matrix.tolist())

        if len(sentence_role_list) == 0:
            print("no srl", sentence, "\n", " ".join(sentence))
            tuple_role_list = []
    
            temp_sentence = list.copy(sentence)
    
            temp_sentence.insert(ac_idx, '<ac>')
            temp_sentence.insert(ace_idx, '</ac>')
    
            tags = ['UNK'] * len(sentence)
    
            for idx, tag in enumerate(temp_sentence):
                if tag in EOS_tokens_list:
                    continue
                tuple_role_list.append((idx, idx, 'UNK'))
    
            if len(tuple_role_list) > 0:
                sentence_role_list.append(tuple_role_list)
            node_num = len(tuple_role_list)
            sentences_srl_role_matrix_list.append(np.ones([node_num, node_num]).tolist())
        # print("sentence_role_list ", sentence_role_list)
        
        # sentence_role_list2 = []
        # for tuple_roles in sentence_role_list:
        #     tuple_role_list = []
        #     for roles in tuple_roles:
        #         s_idx = roles[0]
        #         e_idx = roles[1]
        #         if s_idx >= ac_idx:
        #             s_idx += 1
        #         if e_idx >= ac_idx:
        #             e_idx += 1
        #         tuple_role_list.append((s_idx, e_idx, roles[2]))
        #     sentence_role_list2.append(tuple_role_list)
        
        

        sentences_srl_list.append(sentence_role_list)
    
    # print("sentences_srl_list ", sentences_srl_list)
    para_srl_list.append(sentences_srl_list)

    para_srl_role_matrix_list.append(sentences_srl_role_matrix_list)
                
    # print("******************")

assert len(para_srl_list) == len(orig_AC_spans_list)
assert len(para_srl_role_matrix_list) == len(orig_AC_spans_list)


data_df['para_srl'] = para_srl_list
data_df['para_srl_role_matrix'] = para_srl_role_matrix_list

pe_data_df = pd.DataFrame(data_df)
pe_data_df.to_csv("cdcp_data_df_noabbre_srl.csv", index=False)

print(set(tags_list))

# {'I-ARGM-GOL', 'B-ARGM-MOD', 'B-ARGM-EXT', 'B-ARGM-MNR', 'I-R-ARG1', 'B-R-ARG0', 'B-R-ARG3', 'B-C-ARG2', 'I-ARGM-PRD', 'B-ARGM-CAU', 'B-ARGM-ADV', 'B-ARGM-ADJ', 'B-ARGM-GOL', 'B-C-ARGM-ADV', 'I-ARGM-COM', 'B-R-ARGM-CAU', 'I-ARG3', 'I-C-ARGM-ADV', 'I-ARG2', 'I-C-ARGM-MNR', 'I-ARG1', 'I-ARGM-DIR', 'B-ARG2', 'B-ARGM-DIR', 'I-ARGM-NEG', 'I-ARG0', 'B-R-ARGM-MNR', 'B-ARGM-PRD', 'I-C-ARG2', 'I-ARGM-DIS', 'B-ARG1', 'B-ARGM-REC', 'B-R-ARG1', 'I-R-ARG0', 'B-ARGM-LVB', 'I-ARGM-TMP', 'B-C-ARGM-MNR', 'I-V', 'B-ARG3', 'I-ARGM-CAU', 'B-ARG4', 'I-R-ARGM-LOC', 'I-C-ARG0', 'B-ARGM-COM', 'B-ARG0', 'I-ARGM-EXT', 'B-V', 'B-R-ARGM-TMP', 'B-C-ARG0', 'B-ARGM-PRP', 'I-ARGM-MOD', 'I-ARGM-ADJ', 'I-ARGM-PRP', 'I-R-ARGM-TMP', 'B-C-ARG1', 'B-ARGM-LOC', 'I-C-ARG1', 'I-ARGM-PNC', 'I-C-ARGM-LOC', 'B-R-ARG2', 'O', 'I-R-ARGM-MNR', 'B-ARGM-DIS', 'I-ARG4', 'I-ARGM-MNR', 'B-C-ARGM-LOC', 'I-ARGM-ADV', 'B-ARGM-PNC', 'I-R-ARG3', 'I-ARGM-LOC', 'B-ARGM-TMP', 'B-ARGM-NEG', 'B-R-ARGM-LOC'}


end = time.time()
print(end-start)
        



