import itertools
from collections import namedtuple, defaultdict, Counter
import copy
import argparse
import tqdm
import os
import json
import numpy as np
import os, sys
import datetime
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.basic_utils import get_index_positions
from operator import itemgetter
from os.path import dirname, exists, join, realpath
import subprocess
from easydict import EasyDict as edict

import torch
from utils.basic_utils import save_json, make_zipfile, load_json
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from allennlp.predictors.predictor import Predictor


predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", cuda_device=0)


orig_data_path = "cdcp_data_df.csv"
orig_data_df = pd.read_csv(orig_data_path)
orig_text_list = list(orig_data_df["para_text"])
orig_AC_spans_list = [eval(AC_spans) for AC_spans in orig_data_df['adu_spans']]
# orig_para_srl_list = [eval(para_srl) for para_srl in orig_data_df['para_srl']]

tokenized_text_list = []
new_adu_spans_list = []
new_para_srl_list = []

# special_tokens = {'<pad>', '<essay>', '<para-conclusion>',
#                   '<para-body>', '<para-intro>', '<ac>',
#                   '</essay>', '</para-conclusion>', '</para-body>',
#                   '</para-intro>', '</ac>'}

map = {'cannot': 'can not', 'id': 'identity', 'dont' : "do not", 'wont': 'will not', 'cant': 'can not',
       'youre': 'you are', 'arent': 'are not', 'shes': 'she is', '10pm': '10 pm', '9pm': '9 pm', '9am': '9 am',
        '8pm': '8 pm', '8am': '8 am',
       '7pm': '7 pm', '10M': '10 M', '1692g': '1692 g', '10am': '10 am', '5pm': '5 pm', '12K': '12 K',
       'Thats': 'That is', 'thats': 'that is', 'hes': 'he is', 'doesnt': 'does not', 'im': 'i m'}

para_text_replace_abbre = []
para_adu_spans_noabbre = []
for para_text, AC_spans in zip(orig_text_list, orig_AC_spans_list):
    if len(AC_spans) == 0:
        para_text_replace_abbre.append([])
        continue
    
    orig_pos2bert_pos = {}
    para_tokens = para_text.split(' ')
    # print(para_tokens)
    para_text_list = []
    start = AC_spans[0][0]
    end = AC_spans[-1][1] + 1
    
    
    new_adu_spans_noabbre = []
    count = 0
    
    for index, spans in enumerate(AC_spans):
        s_ind = spans[0]
        e_ind = spans[1]
        
        sentence = para_tokens[spans[0]: spans[1] + 1]
        # print(sentence)
        ac_idx = sentence.index('<ac>')
        ace_idx = sentence.index("</ac>")
        
        assert ac_idx == 0
        assert ace_idx == len(sentence) - 1
        
        sentences_text_list = []
        for word in sentence:
            if word in map.keys():
                sentences_text_list.extend(map[word].split())
            else:
                sentences_text_list.append(word)
        s_ind += count
        count += len(sentences_text_list) - len(sentence)
        e_ind += count
        new_adu_spans_noabbre.append([s_ind, e_ind])

        
        sentences_text_list_ = sentences_text_list.copy()
        
        sentences_text_list.remove('<ac>')
        sentences_text_list.remove('</ac>')
        
        tuples = predictor.predict(sentence=' '.join(sentences_text_list))
        
        for tuple in tuples['verbs']:
            tags = tuple['tags']
            if 'B-ARG' in "".join(tags):
                if len(sentences_text_list) != len(tags):
                    print("length not equal \n", sentence, "\n", tags, "\n", " ".join(sentence))
        
                    print(tuples['words'])
                    print(sentence)
                    print(sentences_text_list)
        
                    inwords_notsent = []
                    insent_notwords = []
                    for word in tuples['words']:
                        if word in sentence:
                            continue
                        else:
                            inwords_notsent.append(word)
                    for word in sentence:
                        if word in tuples['words']:
                            continue
                        else:
                            insent_notwords.append(word)
                    print("inwords_notsent", inwords_notsent)
                    print("insent_notwords", insent_notwords)
        
        
        para_text_list.extend(sentences_text_list_)
        
    para_text_list = para_tokens[:start] + para_text_list + para_tokens[end:]
    
    para_adu_spans_noabbre.append(new_adu_spans_noabbre)
    
    para_text_replace_abbre.append(' '.join(para_text_list))
    


orig_data_df['para_text_replace_abbre'] = para_text_replace_abbre
orig_data_df['adu_spans_noabbre'] = para_adu_spans_noabbre

pe_data_df = pd.DataFrame(orig_data_df)
pe_data_df.to_csv("cdcp_data_df_noabbre.csv", index=False)