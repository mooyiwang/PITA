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
import torch
from models.pos_map_cdcp import pair2sequence


graph_list = []
emo_pos_list = []
cau_pos_list = []
memory = {}
def construct_graph(ac_num, max_dis, memory):
	if ac_num == 0:
		return [], [], [], [], [], [], [], []
	if ac_num == 1:
		return np.ones((1, 2)).tolist(), np.ones((1, 1)).tolist(), [], [], [], [], [], np.ones((3, 2, 2)).tolist()

	if ac_num in memory:
		return memory[ac_num]

	base_idx = np.arange(0, ac_num)
	p1_pos = np.concatenate([base_idx.reshape(-1, 1)] * ac_num, axis=1).reshape(1, -1)[0] # [0 ,0, 0, 0, 0, 1, 1, 1, ...]
	p2_pos = np.concatenate([base_idx] * ac_num, axis=0) # [0 ,1, 2, 3, 4, 0, 1, 2, ...]
	rel_pos = p2_pos - p1_pos
	rel_pos = torch.LongTensor(rel_pos)
	p1_pos = torch.LongTensor(p1_pos)
	p2_pos = torch.LongTensor(p2_pos)

	if ac_num <= max_dis + 1:
		rel_mask = np.array(list(map(lambda x: x > 0, rel_pos.tolist())), dtype=np.int64)
		rel_mask = torch.BoolTensor(rel_mask)
		pair_num = int(sum(rel_mask))

		rel_pos = rel_pos.masked_select(rel_mask)
		p1_pos = p1_pos.masked_select(rel_mask)
		p2_pos = p2_pos.masked_select(rel_mask)
	elif ac_num > max_dis + 1:
		rel_mask = np.array(list(map(lambda x: -max_dis <= x <= max_dis and x > 0, rel_pos.tolist())), dtype=np.int64)
		rel_mask = torch.BoolTensor(rel_mask)
		pair_num = int(sum(rel_mask))

		rel_pos = rel_pos.masked_select(rel_mask)
		p1_pos = p1_pos.masked_select(rel_mask)
		p2_pos = p2_pos.masked_select(rel_mask)
	else:
		raise ValueError

	whole_graph = np.zeros((7, 1 + ac_num + pair_num * 2, 1 + ac_num + pair_num * 2))

	doc_graph = np.ones((1, 1 + ac_num + pair_num * 2))
	actc_graph = np.ones((ac_num, ac_num)) # TODO consider the structure of paragraph
	ari_graph = np.ones((pair_num, pair_num)) # whether needed full connected graph or only a self-loop graph ?
	artc_graph = np.ones((pair_num, pair_num)) # whether needed full connected graph or only a self-loop graph ?
	ac2ari_graph = np.zeros((ac_num, pair_num))
	ac2artc_graph = np.zeros((ac_num, pair_num))

	# TODO whether considering the interaction between ari and artc or not
	ari2artc_graph = np.eye(pair_num) # whether needed only a self-loop graph ?

	count = 0
	# print("p1_pos.tolist()", p1_pos.tolist())
	# print("p2_pos.tolist()", p2_pos.tolist())
	for emo, cau in zip(p1_pos.tolist(), p2_pos.tolist()):
		ac2ari_graph[emo, count] = 1
		ac2ari_graph[cau, count] = 1

		ac2artc_graph[emo, count] = 1
		ac2artc_graph[cau, count] = 1
		# print("emo ", emo)
		# print("cau ", cau)
		# print("count ", count)
		# print(ac2ari_graph.size)
		# print(ac2artc_graph.size)
		count += 1

	whole_graph[0, 0:1, :] = doc_graph
	whole_graph[0, :, 0:1] = doc_graph.T

	whole_graph[1, 1:ac_num + 1, 1:ac_num + 1] = actc_graph

	whole_graph[2, ac_num + 1:ac_num + 1 + pair_num, ac_num + 1:ac_num + 1 + pair_num] = ari_graph

	whole_graph[3, ac_num + 1 + pair_num:ac_num + 1 + pair_num * 2:,
	ac_num + 1 + pair_num:ac_num + 1 + pair_num * 2] = artc_graph

	whole_graph[4, 1:ac_num + 1, ac_num + 1:ac_num + 1 + pair_num] = ac2ari_graph
	whole_graph[4, ac_num + 1:ac_num + 1 + pair_num, 1:ac_num + 1] = ac2ari_graph.T

	whole_graph[5, 1:ac_num + 1:, ac_num + 1 + pair_num:ac_num + 1 + pair_num * 2] = ac2artc_graph
	whole_graph[5, ac_num + 1 + pair_num:ac_num + 1 + pair_num * 2, 1:ac_num + 1:] = ac2artc_graph.T

	whole_graph[6, ac_num + 1:ac_num + 1 + pair_num, ac_num + 1 + pair_num:ac_num + 1 + pair_num * 2] = ari2artc_graph
	whole_graph[6, ac_num + 1 + pair_num:ac_num + 1 + pair_num * 2, ac_num + 1:ac_num + 1 + pair_num] = ari2artc_graph.T

	memory[ac_num] = doc_graph.tolist(), actc_graph.tolist(), ari_graph.tolist(), artc_graph.tolist(), \
	                 ac2ari_graph.tolist(), ac2artc_graph.tolist(), ari2artc_graph.tolist(), whole_graph.tolist()

	return doc_graph.tolist(), actc_graph.tolist(), ari_graph.tolist(), artc_graph.tolist(), \
	       ac2ari_graph.tolist(), ac2artc_graph.tolist(), ari2artc_graph.tolist(), whole_graph.tolist()




def construct_graph6(ac_num, max_dis, memory):
	if ac_num == 0:
		return [], [], [], [], [], [], [], [], [], [], []
	if ac_num == 1:
		return np.ones((1, 3)).tolist(), np.ones((1, 3)).tolist(), [], [], [], [], [], [], [], [], np.ones((3, 3, 3)).tolist()

	if ac_num in memory:
		return memory[ac_num]

	base_idx = np.arange(0, ac_num)
	p1_pos = np.concatenate([base_idx.reshape(-1, 1)] * ac_num, axis=1).reshape(1, -1)[0] # [0 ,0, 0, 0, 0, 1, 1, 1, ...]
	p2_pos = np.concatenate([base_idx] * ac_num, axis=0) # [0 ,1, 2, 3, 4, 0, 1, 2, ...]
	rel_pos = p2_pos - p1_pos
	rel_pos = torch.LongTensor(rel_pos)
	p1_pos = torch.LongTensor(p1_pos)
	p2_pos = torch.LongTensor(p2_pos)

	if ac_num <= max_dis + 1:
		rel_mask = np.array(list(map(lambda x: x > 0, rel_pos.tolist())), dtype=np.int64)
		rel_mask = torch.BoolTensor(rel_mask)
		pair_num = int(sum(rel_mask))

		rel_pos = rel_pos.masked_select(rel_mask)
		p1_pos = p1_pos.masked_select(rel_mask)
		p2_pos = p2_pos.masked_select(rel_mask)
	elif ac_num > max_dis + 1:
		rel_mask = np.array(list(map(lambda x: -max_dis <= x <= max_dis and x > 0, rel_pos.tolist())), dtype=np.int64)
		rel_mask = torch.BoolTensor(rel_mask)
		pair_num = int(sum(rel_mask))

		rel_pos = rel_pos.masked_select(rel_mask)
		p1_pos = p1_pos.masked_select(rel_mask)
		p2_pos = p2_pos.masked_select(rel_mask)
	else:
		raise ValueError

	whole_graph = np.zeros((10, 1 + ac_num + pair_num * 2 + 3, 1 + ac_num + pair_num * 2 + 3))

	doc_graph = np.ones((1, 1 + ac_num + pair_num * 2 + 3))
	t_ac2ac_graph = np.ones((1, 1 + ac_num))
	actc_graph = np.ones((ac_num, ac_num)) # TODO consider the structure of paragraph
	t_ari2ari_graph = np.ones((1, 1 + pair_num))
	ari_graph = np.ones((pair_num, pair_num)) # whether needed full connected graph or only a self-loop graph ?
	t_artc2artc_graph = np.ones((1, 1 + pair_num))
	artc_graph = np.ones((pair_num, pair_num)) # whether needed full connected graph or only a self-loop graph ?
	ac2ari_graph = np.zeros((ac_num, pair_num))
	ac2artc_graph = np.zeros((ac_num, pair_num))

	# TODO whether considering the interaction between ari and artc or not
	ari2artc_graph = np.eye(pair_num) # whether needed only a self-loop graph ?

	count = 0
	# print("p1_pos.tolist()", p1_pos.tolist())
	# print("p2_pos.tolist()", p2_pos.tolist())
	for emo, cau in zip(p1_pos.tolist(), p2_pos.tolist()):
		ac2ari_graph[emo, count] = 1
		ac2ari_graph[cau, count] = 1

		ac2artc_graph[emo, count] = 1
		ac2artc_graph[cau, count] = 1
		# print("emo ", emo)
		# print("cau ", cau)
		# print("count ", count)
		# print(ac2ari_graph.size)
		# print(ac2artc_graph.size)
		count += 1

	whole_graph[0, 0:1, :] = doc_graph
	whole_graph[0, :, 0:1] = doc_graph.T

	whole_graph[1, 1:2, 1:ac_num + 2] = t_ac2ac_graph
	whole_graph[1, 1:ac_num + 2, 1:2] = t_ac2ac_graph.T

	whole_graph[2, 2:ac_num + 2, 2:ac_num + 2] = actc_graph

	whole_graph[3, ac_num + 2:ac_num + 3, ac_num + 2:ac_num + 3 + pair_num] = t_ari2ari_graph
	whole_graph[3, ac_num + 2:ac_num + 3 + pair_num, ac_num + 2:ac_num + 3] = t_ari2ari_graph.T

	whole_graph[4, ac_num + 3:ac_num + 3 + pair_num, ac_num + 3:ac_num + 3 + pair_num] = ari_graph

	whole_graph[5, ac_num + 3 + pair_num:ac_num + 4 + pair_num, ac_num + 3 + pair_num:] = t_artc2artc_graph
	whole_graph[5, ac_num + 3 + pair_num:, ac_num + 3 + pair_num:ac_num + 4 + pair_num] = t_artc2artc_graph.T

	whole_graph[6, ac_num + 4 + pair_num:ac_num + 4 + pair_num * 2, ac_num + 4 + pair_num:ac_num + 4 + pair_num * 2] = artc_graph

	whole_graph[7, 2:ac_num + 2, ac_num + 3:ac_num + 3 + pair_num] = ac2ari_graph
	whole_graph[7, ac_num + 3:ac_num + 3 + pair_num, 2:ac_num + 2] = ac2ari_graph.T

	whole_graph[8, 2:ac_num + 2, ac_num + 4 + pair_num:ac_num + 4 + pair_num * 2] = ac2artc_graph
	whole_graph[8, ac_num + 4 + pair_num:ac_num + 4 + pair_num * 2, 2:ac_num + 2] = ac2artc_graph.T

	whole_graph[9, ac_num + 3:ac_num + 3 + pair_num, ac_num + 4 + pair_num:ac_num + 4 + pair_num * 2] = ari2artc_graph
	whole_graph[9, ac_num + 4 + pair_num:ac_num + 4 + pair_num * 2, ac_num + 3:ac_num + 3 + pair_num] = ari2artc_graph.T

	memory[ac_num] = doc_graph.tolist(), t_ac2ac_graph.tolist(), actc_graph.tolist(), t_ari2ari_graph.tolist(), \
	                 ari_graph.tolist(), t_artc2artc_graph.tolist(), artc_graph.tolist(), ac2ari_graph.tolist(), \
	                 ac2artc_graph.tolist(), ari2artc_graph.tolist(), whole_graph.tolist()

	return doc_graph.tolist(), t_ac2ac_graph.tolist(), actc_graph.tolist(), t_ari2ari_graph.tolist(), \
	       ari_graph.tolist(), t_artc2artc_graph.tolist(), artc_graph.tolist(), ac2ari_graph.tolist(), \
	       ac2artc_graph.tolist(), ari2artc_graph.tolist(), whole_graph.tolist()




if __name__ == '__main__':
	data_df = pd.read_csv("cdcp_data_df2.csv")
	AC_types_list = [list(eval(AC_types)) for AC_types in data_df["ac_types"]]
	AR_pairs_list = [eval(_) for _ in data_df['ac_rel_pairs']]  # ac_types,ac_rel_targets,ac_rel_types,ac_rel_pairs
	# AR_link_list = [eval(_) for _ in data_df['ac_rel_targets']]

	max_ac_num = 28
	max_ar_pair_dis = 12

	# graph variant 1
	doc_graph_list = list()
	actc_graph_list = list()
	ari_graph_list = list()
	artc_graph_list = list()
	ac2ari_graph_list = list()
	ac2artc_graph_list = list()
	ari2artc_graph_list = list()
	whole_graph_list = list()
	for ac_types, ar_pairs in zip(AC_types_list, AR_pairs_list):
		ac_num = len(ac_types)
		doc_grap, actc_grap, ari_grap, artc_grap, ac2ari_grap, ac2artc_grap, ari2artc_grap, whole_grap = construct_graph(ac_num, max_ar_pair_dis, memory)
		doc_graph_list.append(doc_grap)
		actc_graph_list.append(actc_grap)
		ari_graph_list.append(ari_grap)
		artc_graph_list.append(artc_grap)
		ac2ari_graph_list.append(ac2ari_grap)
		ac2artc_graph_list.append(ac2artc_grap)
		ari2artc_graph_list.append(ari2artc_grap)
		whole_graph_list.append(whole_grap)
	data_df['doc_graph'] = doc_graph_list
	data_df['actc_graph'] = actc_graph_list
	data_df['ari_graph'] = ari_graph_list
	data_df['artc_graph'] = artc_graph_list
	data_df['ac2ari_graph'] = ac2ari_graph_list
	data_df['ac2artc_graph'] = ac2artc_graph_list
	data_df['ari2artc_graph'] = ari2artc_graph_list
	data_df['whole_graph'] = whole_graph_list
	
	data_df.to_csv("cdcp_data_df_graphs2.csv", index=False)

	