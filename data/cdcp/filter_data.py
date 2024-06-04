import os
import torch
import sys
import pandas as pd


if __name__ == "__main__":

    orig_data_path = "cdcp_data_df.csv"
    orig_data_df = pd.read_csv(orig_data_path)
    orig_text_list = list(orig_data_df["para_text"])

    truncated_adu_spans_list = list(map(eval, orig_data_df['adu_spans']))
    truncated_ac_types_list = list(map(eval, orig_data_df['ac_types']))
    truncated_ac_rel_pairs_list = list(map(eval, orig_data_df['ac_rel_pairs']))
    truncated_ac_rel_types_list = list(map(eval, orig_data_df['ac_rel_types']))

    droped_ac = 0
    droped_rel = 0

    filted_count = 0
    dis_map = {x: 0 for x in range(13)}
    test_max_dis = 0

    for idx, (para_text, adu_spans, ac_types, ac_rel_pairs, ac_rel_types) \
        in enumerate(zip(orig_data_df['para_text'], truncated_adu_spans_list,
                truncated_ac_types_list, truncated_ac_rel_pairs_list, 
                truncated_ac_rel_types_list)):

        if orig_data_df['dataset_type'][idx] == 'test':
            for pair in ac_rel_pairs:
                dis_map[abs(pair[0] - pair[1])] += 1
                if abs(pair[0] - pair[1]) > test_max_dis:
                    test_max_dis = abs(pair[0] - pair[1])

        if len(ac_types) > 28:
            filted_count += 1
            exceeded_idx = 28
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
    orig_data_df['adu_spans'] = truncated_adu_spans_list
    orig_data_df['ac_types'] = truncated_ac_types_list
    orig_data_df['ac_rel_pairs'] = truncated_ac_rel_pairs_list
    orig_data_df['ac_rel_types'] = truncated_ac_rel_types_list

    # orig_data_df.to_csv("cdcp_data_df2.csv", index=False)

    print('filted ac: {}'.format(filted_count))
    print('droped ac: {}'.format(droped_ac))
    print('droped rel: {}'.format(droped_rel))

    print('test max dis: {}'.format(test_max_dis))
    print(dis_map)



