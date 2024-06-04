import os, time
import pandas as pd
import numpy as np


data_path= "./data/cdcp/cdcp_data_df_srl2.csv"
data_df = pd.read_csv(data_path)
data_df = data_df[data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]

para_text_list = list(data_df['para_text'])
orig_AC_spans_list = [eval(AC_spans) for AC_spans in data_df['adu_spans']]
orig_para_srl_list = [eval(para_srl) for para_srl in data_df['para_srl']]


for index, para_srl_list in enumerate(orig_para_srl_list):
	para_srl_for_bert = []
	
count = 0
for para_text, AC_spans, para_srl_list in zip(para_text_list, orig_AC_spans_list, orig_para_srl_list):
	para_tokens = para_text.split(' ')
	for sentence_srl_list, spans in zip(para_srl_list, AC_spans):
		if len(sentence_srl_list) == 0:
			print(count, para_text, spans)
			count += 1
		
		sentence = para_tokens[spans[0]: spans[1]+1]
		ac_idx = sentence.index('<ac>')
		
		if ac_idx != 0:
			print(spans, para_text, sentence)