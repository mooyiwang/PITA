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

data_df = pd.read_csv("cdcp_data_df2.csv")

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
max_ac_num = max([len(list(eval(AC_types))) for AC_types in data_df["ac_types"]])
print("max_ac_num", max_ac_num)

AR_pairs_list = [eval(_) for _ in data_df['ac_rel_pairs']] # ac_types,ac_rel_targets,ac_rel_types,ac_rel_pairs

max_ac_num = 0
max_ar_pair_dis = 0
num_map = {x: 0 for x in range(34)}
dis_map = {x: 0 for x in range(13)}
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
# {0: 0, 1: 0, 2: 101, 3: 118, 4: 105, 5: 76, 6: 62, 7: 54, 8: 37, 9: 33, 10: 29, 11: 21, 12: 20, 13: 9, 14: 18, 15: 11,
# 16: 5, 17: 5, 18: 4, 19: 2, 20: 3, 21: 1, 22: 2, 23: 4, 24: 3, 25: 2, 26: 1, 27: 2, 28: 1, 29: 0, 30: 1, 31: 0, 32: 0, 33: 1}
# {0: 0, 1: 943, 2: 248, 3: 86, 4: 36, 5: 14, 6: 5, 7: 5, 8: 6, 9: 4, 10: 3, 11: 1, 12: 2}


# {0: 0, 1: 0, 2: 101, 3: 118, 4: 105, 5: 76, 6: 62, 7: 54, 8: 37, 9: 33, 10: 29, 11: 21, 12: 20, 13: 9, 14: 18, 15: 11,
# 16: 5, 17: 5, 18: 4, 19: 2, 20: 3, 21: 1, 22: 2, 23: 4, 24: 3, 25: 2, 26: 1, 27: 2, 28: 3}
# {0: 0, 1: 942, 2: 248, 3: 86, 4: 36, 5: 14, 6: 5, 7: 5, 8: 6, 9: 4, 10: 3, 11: 1, 12: 2}


# {0: 0, 1: 195, 2: 41, 3: 16, 4: 7, 5: 4, 6: 2, 7: 2, 8: 2, 9: 1, 10: 0, 11: 1, 12: 1}