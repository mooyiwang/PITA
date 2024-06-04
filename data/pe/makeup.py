from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import nltk
from pathlib import Path
import os, time
import pandas as pd
import numpy as np



s1 = "since <ac> bicycle expels neither carbon dioxides nor harmful gasses </ac>"
s1_tags = ["B-ARGM-CAU",
        "O",
        "B-ARG0",
        "B-V",
        "B-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
	"O"
      ]

s2 = "since <ac> youngsters , as the target consumers , love celebrities </ac>"
s2_tags = [
        "B-ARGM-CAU",
        "O",
        "B-ARG0",
        "O",
        "B-ARGM-PRD",
        "I-ARGM-PRD",
        "I-ARGM-PRD",
        "I-ARGM-PRD",
        "O",
        "B-V",
        "B-ARG1",
	"O"
      ]

s3 = "for instance , <ac> lions hunt in a team </ac>"
s3_tags = [
        "B-ARGM-DIS",
        "I-ARGM-DIS",
        "O",
	"O",
        "B-ARG0",
        "B-V",
        "B-ARGM-MNR",
        "I-ARGM-MNR",
        "I-ARGM-MNR",
	"O"
      ]

s4 = "in contrast , <ac> manual harvest prevents severe environmental damages </ac>"
s4_tags = [
        "B-ARGM-DIS",
        "I-ARGM-DIS",
        "O",
		"O",
        "B-ARG0",
        "B-V",
        "B-ARG1",
        "I-ARG1",
        "I-ARG1",
		"O"
      ]

s5 = "as well as <ac> it ' s one way of recent communications </ac>"
s5_tags = [
        "B-ARGM-DIS",
        "I-ARGM-DIS",
		"I-ARGM-DIS",
		"O",
        "B-ARG1",
        "B-V",
        "B-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
	"O",
      ]


s6 = "<ac> the condition of job market changes enormously fast like a flash </ac>"
s6_tags = [	"O",
        "B-ARG1",
	"I-ARG1",
	"I-ARG1",
	"I-ARG1",
	"I-ARG1",
        "B-V",
        "B-ARGM-MNR",
        "I-ARGM-MNR",
        "B-ARGM-MNR",
        "I-ARGM-MNR",
        "I-ARGM-MNR",
	"O"
      ]


s7 = "but <ac> i don ' t like electronics </ac>"
s7_tags = [
        "B-ARGM-DIS",
	"O",
        "B-ARG0",
        "O",
	"O",
        "B-ARGM-NEG",
        "B-V",
        "B-ARG1",
	"O",
      ]


s8 = "because <ac> the change of job , not the same task day by day </ac>"
s8_tags = [
        "B-ARGM-CAU",
		"O",
        "B-ARGM-CAU",
        "I-ARGM-CAU",
        "I-ARGM-CAU",
        "I-ARGM-CAU",
        "O",
	"B-ARGM-NEG"
        "B-ARG1",
        "I-ARG1",
        "I-ARG1",
        "B-ARG2",
        "I-ARG2",
        "I-ARG2",
	"O"
      ]

s9 = "<ac> the more money , the more chances </ac>"
s9_tags = [
        "B-ARG0",
        "I-ARG0",
        "I-ARG0",
        "O",
        "B-ARG1",
        "I-ARG1",
        "I-ARG1",
      ]


s10 = "<ac> creativity originates in a free mind </ac>"
s10_tags = [	"O",
        "B-ARG1",
        "B-V",
        "B-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
	"O"
      ]


s11 = "<ac> harmful chemicals present in industrial waste kills lot of sea creatures in oceans or rivers </ac>"
s11_tags = [	"O",
        "B-ARG0",
        "I-ARG0",
        "I-ARG0",
        "I-ARG0",
	"I-ARG0",
	"I-ARG0",
        "B-V",
        "B-ARG1",
	"I-ARG1",
	"I-ARG1",
	"I-ARG1",
        "B-ARGM-LOC",
        "I-ARGM-LOC",
        "I-ARGM-LOC",
        "I-ARGM-LOC",
	"O"
      ]

s12 = "first and foremost , <ac> email can be count as one of the most beneficial results of modern technology </ac>"
s12_tags = [
        "B-ARGM-ADV",
        "I-ARGM-ADV",
        "I-ARGM-ADV",
        "O",
	"O",
        "B-ARG1",
	"O",
        "O",
        "B-V",
        "B-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
        "I-ARG2",
	"O",
      ]


s13 = "<ac> likewise , other animals such as , horses , cattles and donkeys for farming , agriculture and as a mode of transportation </ac>"
s13_tags = [	"O",
        "B-ARGM-DIS",
        "O",
        "B-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
        "B-V",
        "B-ARG2",
        "I-ARG2",
        "I-ARG2",
        "O",
        "B-V",
        "B-ARG3",
        "I-ARG3",
        "I-ARG3",
        "I-ARG3",
	"O"
      ]

s14 = "<ac> and above all pleasure hunting of defenseless animals </ac>"
s14_tags = [	"O",
        "B-ARGM-DIS",
        "B-ARG0",
	"I-ARG0",
        "B-V",
        "B-ARG1",
        "I-ARG1",
        "I-ARG1",
        "I-ARG1",
	"O"
      ]


s15 = "<ac> greenhouse effect causes of climate change </ac>"
s15_tags = [	"O",
        "B-ARG0",
	"I-ARG0",
        "B-V",
        "B-ARG1",
        "I-ARG1",
        "I-ARG1"
      ]


s16 = "<ac> this , by all means , is outweigh by youngsters ' adventurousness and bravery </ac>"
s16_tags = [	"O",
	"B-ARG0",
	"O",
        "B-ARGM-ADV",
        "I-ARGM-ADV",
        "I-ARGM-ADV",
        "O",
	"O",
        "B-V",
        "B-ARG1",
	"I-ARG1",
	"I-ARG1",
	"I-ARG1",
	"I-ARG1",
	"O"
      ]


s17 = "for example <ac> playgrounds , parks , science museums , cinemas etc </ac>"
s17_tags = [
        "B-ARGM-DIS",
	"I-ARGM-DIS",
	"O",
        "B-ARG0",
	"O",
        "B-ARG1",
        "O",
        "B-ARG2",
        "I-ARG2",
	"O",
	"B-ARG3",
        "I-ARG3",
	"O"
      ]


s18 = "finally , <ac> in big cities children access to media easily </ac>"
s18_tags = [
        "B-ARGM-TMP",
        "O",
	"O",
	"B-ARGM-LOC",
        "I-ARGM-LOC",
        "I-ARGM-LOC"
        "B-ARG0",
        "B-V",
	"O"
        "B-ARG1",
        "B-ARGM-MNR",
	"O"
      ]


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




role_chars = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'R-ARG0','R-ARG1','R-ARG2','R-ARG3','R-ARG4','C-ARG0','C-ARG1','C-ARG2','C-ARG3', 'C-ARG4',
              'V','DIS', 'ADJ', 'TMP', 'CAU', 'MNR', 'EXT', 'ADV', 'CAU', 'LOC', 'PRP', 'NEG', 'PRD', 'COM',
              'GOL', 'DIR', 'PNC']


mu_tags = [s1_tags, s2_tags, s3_tags, s4_tags, s5_tags, s6_tags, s7_tags, s8_tags, s9_tags, s10_tags,
        s11_tags, s12_tags, s13_tags, s14_tags, s15_tags, s16_tags, s17_tags, s18_tags]


sentences_srl_list = []
sentences_srl_role_matrix_list = []
for i in range(0, 18):
	tags = mu_tags[i]
	pre_ind = 0
	pre_role = ''
	sentence_role_list = []
	tuple_role_list = []
	for idx, tag in enumerate(tags):
		if pre_role != '' and pre_role != tag[2:]:
			if ("ARG" in pre_role or "V" in pre_role) and pre_role.split("-")[-1] in role_chars:
				tuple_role_list.append((pre_ind, idx -1, pre_role))
			pre_ind = idx
			pre_role = tag[2:]
		
		if tag == 'O':
			pre_ind = idx + 1
			pre_role = ''
			continue
		
		if pre_role == '' and pre_role != tag[2:]:
			pre_role = tag[2:]
		
		elif pre_role != '' and pre_role == tag[2:]:
			if idx == len(tags ) -1:
				if ("ARG" in pre_role or "V" in pre_role) and pre_role.split("-")[-1] in role_chars:
					tuple_role_list.append((pre_ind, idx, pre_role))
					
	sentence_role_list.append(tuple_role_list)
	
	sentences_srl_list.append(sentence_role_list)
	
	sentences_srl_role_matrix_list.append(get_interaction_matrix(sentence_role_list).tolist())
	

print(sentences_srl_list)
print(sentences_srl_role_matrix_list)


sentences_srl_list = [[[(0, 0, 'ARGM-CAU'), (2, 2, 'ARG0'), (3, 3, 'V'), (4, 9, 'ARG1')]],# (98, 108)
                      [[(0, 0, 'ARGM-CAU'), (2, 2, 'ARG0'), (4, 7, 'ARGM-PRD'), (9, 9, 'V'), (10, 10, 'ARG1')]], # (92, 103)
                      [[(0, 1, 'ARGM-DIS'), (4, 4, 'ARG0'), (5, 5, 'V'), (6, 8, 'ARGM-MNR')]], # (56, 65)
                      [[(0, 1, 'ARGM-DIS'), (4, 4, 'ARG0'), (5, 5, 'V'), (6, 8, 'ARG1')]], # (37, 47)
                      [[(0, 2, 'ARGM-DIS'), (4, 4, 'ARG1'), (5, 5, 'V'), (6, 10, 'ARG2')]], # (37, 49)
                      [[(1, 5, 'ARG1'), (6, 6, 'V'), (7, 11, 'ARGM-MNR')]], # (68, 80)
                      [[(0, 0, 'ARGM-DIS'), (2, 2, 'ARG0'), (5, 5, 'ARGM-NEG'), (6, 6, 'V'), (7, 7, 'ARG1')]], # (25, 33)
                      [[(0, 0, 'ARGM-CAU'), (2, 5, 'ARGM-CAU'), (7, 7, 'ARGM-NEGB-ARG1'), (8, 9, 'ARG1'), (10, 12, 'ARG2')]], # (127, 141)
                      [[(0, 2, 'ARG0'), (4, 6, 'ARG1')]], # (83, 91)
                      [[(1, 1, 'ARG1'), (2, 2, 'V'), (3, 6, 'ARG2')]], # (22, 29)
                      [[(1, 6, 'ARG0'), (7, 7, 'V'), (8, 11, 'ARG1'), (12, 15, 'ARGM-LOC')]], # (42, 58)
                      [[(0, 2, 'ARGM-ADV'), (5, 5, 'ARG1'), (8, 8, 'V'), (9, 18, 'ARG2')]], # (1, 20)
                      [[(1, 1, 'ARGM-DIS'), (3, 12, 'ARG1'), (13, 13, 'V'), (14, 16, 'ARG2'), (18, 18, 'V'), (19, 22, 'ARG3')]], # (27, 50)
                      [[(1, 1, 'ARGM-DIS'), (2, 3, 'ARG0'), (4, 4, 'V'), (5, 8, 'ARG1')]], # (122, 131)
                      [[(1, 2, 'ARG0'), (3, 3, 'V'), (4, 6, 'ARG1')]], # (38, 45)
                      [[(1, 1, 'ARG0'), (3, 5, 'ARGM-ADV'), (8, 8, 'V'), (9, 13, 'ARG1')]], # (44, 59)
                      [[(0, 1, 'ARGM-DIS'), (3, 3, 'ARG0'), (5, 5, 'ARG1'), (7, 8, 'ARG2'), (10, 11, 'ARG3')]], # (35, 47)
                      [[(0, 0, 'ARGM-TMP'), (3, 4, 'ARGM-LOC'), (5, 5, 'ARGM-LOCB-ARG0'), (6, 6, 'V'), (7, 7, '-ARG1'), (8, 8, 'ARGM-MNR')]]
                      ] # (1, 12)

sentences_srl_role_matrix_list = [
	[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0], [1.0, 1.0]],
	[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
	[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
]
