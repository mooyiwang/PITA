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
import nltk
from allennlp.predictors.predictor import Predictor
import time


start = time.time()

# predictor = Predictor.from_path(
# 	"https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
# 	cuda_device=0)

data_path = "./data/cdcp/cdcp_data_df.csv"
data_df = pd.read_csv(data_path)

# data_df = data_df[data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]
para_text_list = list(data_df['para_text'])
orig_AC_spans_list = [eval(AC_spans) for AC_spans in data_df['adu_spans']]

special_tokens = {'<pad>', '<essay>', '<para-conclusion>',
                  '<para-body>', '<para-intro>', '<ac>',
                  '</essay>', '</para-conclusion>', '</para-body>',
                  '</para-intro>', '</ac>'}

AC_spans_list = []
tags_list = []

role_chars = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'C-ARG0',
              'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4',
              'V', 'DIS', 'ADJ', 'TMP', 'CAU', 'MNR', 'EXT', 'ADV', 'LOC', 'PRP', 'NEG', 'PRD', 'COM',
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

EOS_tokens_list = [".", "!", "?", "_", '<ac>', '</ac>']


para_srl_list = []
para_srl_role_matrix_list = []

all_inwords_notsent = []
all_insent_notwords = []
# for para_text, AC_spans in zip(para_text_list, orig_AC_spans_list):
# 	if len(AC_spans) == 0:
# 		para_srl_list.append([])
# 		para_srl_role_matrix_list.append([])
# 		continue
#
# 	orig_pos2bert_pos = {}
# 	para_tokens = para_text.split(' ')
# 	# print(para_tokens)
# 	sentences_srl_list = []
# 	sentences_srl_role_matrix_list = []
#
# 	for index, spans in enumerate(AC_spans):
# 		sentence = para_tokens[spans[0]: spans[1] + 1]
# 		# print(sentence)
# 		ac_idx = sentence.index('<ac>')
# 		ace_idx = sentence.index("</ac>")
#
# 		assert ac_idx == 0
# 		assert ace_idx == len(sentence) - 1
#
# 		sentence.remove('<ac>')
# 		sentence.remove('</ac>')
# 		# print(sentence)
#
# 		# print(sentence)
# 		tuples = predictor.predict(sentence=" ".join(sentence))
# 		sentence_role_list = []
# 		# print(tuples)
#
# 		for tuple in tuples['verbs']:
# 			tags = tuple['tags']
# 			if 'B-ARG' in "".join(tags):
# 				# print(tags)
# 				tags_list.extend(tags)
# 				if len(sentence) != len(tags):
# 					print("length not equal \n", sentence, "\n", tags, "\n", " ".join(sentence))
#
# 					print(tuples['words'])
# 					print(sentence)
#
# 					inwords_notsent = []
# 					insent_notwords = []
# 					for word in tuples['words']:
# 						if word in sentence:
# 							continue
# 						else:
# 							inwords_notsent.append(word)
# 					for word in sentence:
# 						if word in tuples['words']:
# 							continue
# 						else:
# 							insent_notwords.append(word)
# 					print("inwords_notsent", inwords_notsent)
# 					print("insent_notwords", insent_notwords)
#
# 					all_inwords_notsent.append(inwords_notsent)
# 					all_insent_notwords.append(insent_notwords)
#
# print(all_inwords_notsent)
# print(all_insent_notwords)
#
# print(set([word for sent in all_inwords_notsent for word in sent]))
# print(set([word for sent in all_insent_notwords for word in sent]))


a = [['can'], ['i', 'd'],
     ['can', 'not'], ['do', 'nt'],
     ['does', 'nt'], ['wo', 'nt'],
     ['ca', 'nt', 'do', 'nt'], ['re'], ['do', 'nt', 'do', 'nt'],['are', 'nt'],
     ['she', 's'],  ['10', 'pm', '9', 'pm'],
     ['9', 'am', '7', 'pm'],
     ['10', 'M'], ['can'], ['1692', 'g'], ['10', 'am'], ['5', 'pm'], ['not'], ['12', 'K'],
     ['That', 's', 'i', 'm'],
     ['ca', 'nt'], ['that', 's'],
     ['ca', 'nt', 's'], ['7', 'pm'],
     ['9', 'am'], ['9', 'am', '7', 'pm'], ['8', 'pm'], ['8', 'am', '9', 'pm'], ['8', 'pm']]

b = [['cannot'], ['id'], ['dont'], ['doesnt'], ['wont'], ['cant', 'dont'], ['youre'], ['youre'], ['dont', 'dont'], ['arent'], ['shes'], ['10pm', '9pm'],
                 ['9am', '7pm'], ['10M'], ['1692g'], ['10am'], ['5pm'], ['12K'],
                 ['Thats', 'im'], ['cant'], ['thats'], ['cant', 'hes'], ['7pm'], ['9am'], ['8pm'], ['8am', '9pm']]

map = {'cannot': 'can not', 'id': 'identity', 'dont' : "do not", 'wont': 'will not', 'cant': 'can not',
       'youre': 'you are', 'arent': 'are not', 'shes': 'she is', '10pm': '10 pm', '9pm': '9 pm',
       '7pm': '7 pm', '10M': '10 M', '1692g': '1692 g', '10am': '10 am', '5pm': '5 pm', '12K': '12 K',
       'Thats': 'That is', 'hes': 'he is'}

print(set(a))
print(set(b))

# {'am', '8', 'm', '7', 's', 'do', '10', '5', 're', 'ca', 'not', '9', 'can', 'she', '1692', 'M', 'wo', 'That', 'pm', '12', 'does', 'd', 'i', 'that', 'g', 'nt', 'are', 'K'}
# {'youre', 'cannot', '12K', '9am', '10M', 'dont', 'id', 'thats', 'doesnt', 'hes', 'im', '1692g', '9pm', '5pm', '8am', 'Thats', '8pm', 'wont', '10pm', '7pm', 'shes', 'arent', '10am', 'cant'}


# 161,541,<para-body> <ac> I dont mind the spouse . </ac> <ac> I mean shes my wife </ac> <ac> so I dont see why not . </ac> <ac> But I dont agree with girlfriends or boyfriends for those who dont have a spouse . </ac> <ac> Spouse is okay . </ac> </para-body>,"[(1, 8), (9, 15), (16, 24), (25, 42), (43, 48)]","['value', 'fact', 'value', 'value', 'value']","[(0, 1), (2, 1)]","['reason', 'reason']",train,body,<para-body> <ac> I do not mind the spouse . </ac> <ac> I mean she is my wife </ac> <ac> so I do not see why not . </ac> <ac> But I do not agree with girlfriends or boyfriends for those who do not have a spouse . </ac> <ac> Spouse is okay . </ac> </para-body>,"[[1, 9], [10, 17], [18, 27], [28, 47], [48, 53]]"
