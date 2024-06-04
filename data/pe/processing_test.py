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


single_ac_type = set()

data_df = pd.read_csv("pe_data_df.csv")

# ac_types_list = data['ac_types']
# mc_count = 0
# c_count = 0
# for ac_types in ac_types_list:
# 	ac_types = eval(ac_types)
# 	if len(ac_types) == 1:
# 		single_ac_type.add(ac_types[0])
# 		if ac_types[0] == 'MajorClaim':
# 			mc_count += 1
# 		else:
# 			c_count += 1
# print(single_ac_type)
# print(mc_count, c_count)
# {'MajorClaim', 'Claim'}
# 375 1


# 检查每一对具有关系的AC之间的距离是多少】
AC_types_list = [list(eval(AC_types)) for AC_types in data_df["ac_types"]]
AR_pairs_list = [eval(_) for _ in data_df['ac_rel_pairs']] # ac_types,ac_rel_targets,ac_rel_types,ac_rel_pairs
AR_link_list = [eval(_) for _ in data_df['ac_rel_targets']]

max_ac_num = 0
max_ar_pair_dis = 0
num_map = {x: 0 for x in range(13)}
dis_map = {x: 0 for x in range(12)}
for ac_types, ar_pairs in zip(AC_types_list, AR_pairs_list):
    if len(ac_types) > max_ac_num:
        max_ac_num = len(ac_types)
    num_map[len(ac_types)] += 1

    for pair in ar_pairs:
        dis_map[abs(pair[0] - pair[1])] += 1
        if abs(pair[0] - pair[1]) > max_ar_pair_dis:
            max_ar_pair_dis = abs(pair[0] - pair[1])
            # if abs(pair[0] - pair[1]) == 9:
            #     print(ar_pairs)
print(max_ac_num)
print(max_ar_pair_dis)
# 12
# 11

print(num_map)
print(dis_map)