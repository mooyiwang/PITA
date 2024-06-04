import os
import ujson as json
import zipfile
import numpy as np
import pickle
import torch
import scipy.sparse as sp
import torch.nn as nn
from models.pos_map_cdcp import pair_idx_map, pair2sequence, pair_num_map


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def concat_json_list(filepaths, save_path):
    json_lists = []
    for p in filepaths:
        json_lists += load_json(p)
    save_json(json_lists, save_path)


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def convert_to_seconds(hms_time):
    """ convert '00:01:12' to 72 seconds.
    :hms_time (str): time in comma separated string, e.g. '00:01:12'
    :return (int): time in seconds, e.g. 72
    """
    times = [float(t) for t in hms_time.split(":")]
    return times[0] * 3600 + times[1] * 60 + times[2]


def get_video_name_from_url(url):
    return url.split("/")[-1][:-4]


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_dirs=None, exclude_extensions=None,
                 exclude_dirs_substring=None):
    """make a zip file of root_dir, save it to save_path.
    exclude_paths will be excluded if it is a subdir of root_dir.
    An enclosing_dir is added is specified.
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as zf:
        for dirname, subdirs, files in os.walk(src_dir):
            if exclude_dirs is not None:
                for e_p in exclude_dirs:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            if exclude_dirs_substring is not None:
                to_rm = []
                for d in subdirs:
                    if exclude_dirs_substring in d:
                        to_rm.append(d)
                for e in to_rm:
                    subdirs.remove(e)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1:])
            zf.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1:])
                zf.write(absname, arcname)


class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dissect_by_lengths(np_array, lengths, dim=0, assert_equal=True):
    """Dissect an array (N, D) into a list a sub-array,
    np_array.shape[0] == sum(lengths), Output is a list of nd arrays, singlton dimention is kept"""
    if assert_equal:
        assert len(np_array) == sum(lengths)
    length_indices = [0, ]
    for i in range(len(lengths)):
        length_indices.append(length_indices[i] + lengths[i])
    if dim == 0:
        array_list = [np_array[length_indices[i]:length_indices[i+1]] for i in range(len(lengths))]
    elif dim == 1:
        array_list = [np_array[:, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    elif dim == 2:
        array_list = [np_array[:, :, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    else:
        raise NotImplementedError
    return array_list


def get_ratio_from_counter(counter_obj, threshold=200):
    keys = counter_obj.keys()
    values = counter_obj.values()
    filtered_values = [counter_obj[k] for k in keys if k > threshold]
    return float(sum(filtered_values)) / sum(values)


def get_rounded_percentage(float_number, n_floats=2):
    return round(float_number * 100, n_floats)


def get_edge_frompairs(group_pairs_list, span_num, type_label=None, value=0):
    # out_matrix = torch.zeros(len(group_pairs_list), span_num, span_num).cuda()
    out_matrix = np.ones([len(group_pairs_list), span_num, span_num]) * value
    tril_matrix = []
    triu_matrix = []
    scr_idx, tar_idx = np.triu_indices(span_num, k=1)
    for idx, pairs in enumerate(group_pairs_list):

        for n, pair in enumerate(pairs):
            if type_label != None:
                out_matrix[idx, min(pair[0], pair[1]), max(pair[0], pair[1])] = type_label[idx][n]
            else:
                out_matrix[idx, min(pair[0], pair[1]), max(pair[0], pair[1])] = 1

        triu_matrix.append(out_matrix[idx, scr_idx, tar_idx])
        # tril_idx = np.tril_indices(span_num, k=-1)
        # tril_matrix.append(out_matrix[idx, tar_idx, scr_idx])

    # out_matrix = torch.Tensor(out_matrix).cuda() # [group_size, span_num, span_num]
    triu_matrix = torch.Tensor(triu_matrix).view(-1)  # [group_size * direct_edge]
    # tril_label = torch.Tensor(tril_matrix).cuda().view(-1)  # [group_size * direct_edge]

    return triu_matrix.tolist()


def get_tuple_frompairs(rel_pair_list, rel_label_list):
    tuple = []
    for rel_pairs, rel_label in zip(rel_pair_list, rel_label_list):
        cur_tuple = []
        if isinstance(rel_label, torch.Tensor):
            rel_label = rel_label.tolist()
        # if isinstance(rel_pairs, torch.Tensor):
        #     rel_pairs = rel_pairs.tolist()
        assert len(rel_pairs) == len(rel_label)
        for pair, label in zip(rel_pairs, rel_label):
            cur_tuple.append((min(pair[0], pair[1]), max(pair[0], pair[1]), label))
        tuple.append(cur_tuple)
    return tuple


def _pair2sequence(pair_list, span_num):
    pairs_true = []
    pairs_num = len(pair_idx_map[span_num])
    for i in range(len(pair_list)):
        true_indices = [pair2sequence[span_num].index((min(x), max(x))) for x in pair_list[i]]
        temp = torch.zeros(pairs_num)
        temp[true_indices] = 1.
        pairs_true.append(temp)
    pairs_true = torch.FloatTensor(torch.cat(pairs_true)).cuda()
    return pairs_true


def _sequence2pair(x, span_num, group_size, value=0):
    x = x.view(group_size, -1)
    # print("x", x.size(), span_num, pair_num_map[span_num])
    assert x.size() == (group_size, pair_num_map[span_num]), print(x.size(), "\n", (group_size, pair_num_map[span_num]))
    x = x.tolist()
    scr_idx, tar_idx = np.triu_indices(span_num, k=1)
    out_matrix = np.ones([group_size, span_num, span_num]) * value
    triu_matrix = []
    pair_list = []
    for idx in range(group_size):
        cur_pairs = []
        for n, pair in enumerate(pair2sequence[span_num]):
            out_matrix[idx, pair[0], pair[1]] = x[idx][n]
            if x[idx][n] == 1 and value == 0:
                cur_pairs.append(pair)
        triu_matrix.append(out_matrix[idx, scr_idx, tar_idx])
        pair_list.append(cur_pairs)
    triu_matrix = torch.Tensor(triu_matrix).view(-1).tolist()  # [group_size * direct_edge]
    return triu_matrix, pair_list


class Scorer:
    def __init__(self):
        self.s = 0
        self.g = 0
        self.c = 0
        return

    def add(self, predict, gold):
        self.s += len(predict)
        self.g += len(gold)
        self.c += len(gold & predict)
        return

    @property
    def p(self):
        return self.c / self.s if self.s else 0.

    @property
    def r(self):
        return self.c / self.g if self.g else 0.

    @property
    def f(self):
        p = self.p
        r = self.r
        return (2. * p * r) / (p + r) if p + r > 0 else 0.0

    def dump(self):
        return {
            'g': self.g,
            's': self.s,
            'c': self.c,
            'p': self.p,
            'r': self.r,
            'f': self.f
        }

def eval_edge(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[2] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 2, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "Support" , 1: "Attack"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in p_sample if edge[2] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in g_sample if edge[2] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    # link_scorer = Scorer()
    # for p_sample, g_sample in zip(predict_list, gold_list):
    #     link_scorer.add(
    #         predict=set([
    #             (
    #                 edge[0],
    #                 edge[1]
    #             )
    #             for edge in p_sample
    #         ]),
    #         gold=set([
    #             (
    #                 edge[0],
    #                 edge[1]
    #             )
    #             for edge in g_sample
    #         ]),
    #     )
    # label_scores['link'] = link_scorer.dump()
    #
    # scorer = Scorer()
    # for p_sample, g_sample in zip(predict_list, gold_list):
    #     scorer.add(
    #         predict=set([
    #             (
    #                 edge[0],
    #                 edge[1],
    #                 edge[2]
    #             )
    #             for edge in p_sample
    #         ]),
    #         gold=set([
    #             (
    #                 edge[0],
    #                 edge[1],
    #                 edge[2]
    #             )
    #             for edge in g_sample
    #         ]),
    #     )
    # label_scores['total'] = scorer.dump()
    return label_scores


def eval_edge_cdcp(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[2] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 2, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "reason" , 1: "evidence"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in p_sample if edge[2] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in g_sample if edge[2] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores


def args_metric(true_args_list, pred_args_list):
    tp, tn, fp, fn = 0, 0, 0, 0
    for true_args, pred_args in zip(true_args_list, pred_args_list):
        true_args_set = set(true_args)
        pred_args_set = set(pred_args)
        assert len(true_args_set) == len(true_args)
        assert len(pred_args_set) == len(pred_args)
        tp += len(true_args_set & pred_args_set)
        fp += len(pred_args_set - true_args_set)
        fn += len(true_args_set - pred_args_set)
    if tp + fp == 0:
        pre = tp/(tp + fp + 1e-10)
    else:
        pre = tp/(tp + fp)
    if tp + fn == 0:
        rec = tp/(tp + fn + 1e-10)
    else:
        rec = tp/(tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre * rec)/(pre + rec + 1e-10)
    else:
        f1 = (2 * pre * rec)/(pre + rec)
    acc = (tp + tn)/(tp + tn + fp + fn + 1e-10)
    return {'pre': pre, 'rec': rec, 'f1': f1, 'acc': acc}


def get_eval_result(res_dict):
    ARI_msg = 'ARI-Macro: {:.4f}\tRel: {:.4f}\tNo-Rel: {:.4f}'.format(
            res_dict["ARI-Macro"], res_dict["Rel"], res_dict["No-Rel"])
    ACTC_msg = 'ACTC-Macro: {:.4f}\tMC: {:.4f}\tClaim: {:.4f}\tPremise: {:.4f}'.format(
            res_dict["ACTC-Macro"], res_dict["MC"], res_dict["Claim"], res_dict["Premise"])
    ARTC_msg = 'ARTC-Macro: {:.4f}\tSup: {:.4f}\tAtc: {:.4f}'.format(
        res_dict["ARTC-Macro"], res_dict["Sup"], res_dict["Atc"])
    ARTC_msg2 = 'ARTC-F1: {:.4f}\tPre: {:.4f}\tRec: {:.4f}'.format(
        res_dict["ARTC-F1"], res_dict["Pre"], res_dict["Rec"])
    macro_msg = 'Total-Macro: {:.4f}\tARI-Macro: {:.4f}\tACTC-Macro: {:.4f}\tARTC-Macro: {:.4f}'.format(
            res_dict["Total-Macro"], res_dict["ARI-Macro"], res_dict["ACTC-Macro"], res_dict["ARTC-Macro"])
    return ARI_msg, ACTC_msg, ARTC_msg, ARTC_msg2, macro_msg


def get_cdcp_eval_result(res_dict):
    ARI_msg = 'ARI-Macro: {:.4f}\tRel: {:.4f}\tNo-Rel: {:.4f}'.format(
            res_dict["ARI-Macro"], res_dict["Rel"], res_dict["No-Rel"])
    ACTC_msg = 'ACTC-Macro: {:.4f}\tvalue: {:.4f}\tpolicy: {:.4f}\ttestimony: {:.4f}\tfact: {:.4f}\treference: {:.4f}'.format(
            res_dict["ACTC-Macro"], res_dict["value"], res_dict["policy"], res_dict["testimony"], res_dict["fact"], res_dict["reference"])
    ACTC_msg2 = 'ACTC-F1: {:.4f}'.format(res_dict["ACTC-F1"])
    ARTC_msg = 'ARTC-Macro: {:.4f}\treason: {:.4f}\tevidence: {:.4f}'.format(
        res_dict["ARTC-Macro"], res_dict["reason"], res_dict["evidence"])
    ARTC_msg2 = 'ARTC-F1: {:.4f}\tPre: {:.4f}\tRec: {:.4f}'.format(
        res_dict["ARTC-F1"], res_dict["Pre"], res_dict["Rec"])
    macro_msg = 'Total-Macro: {:.4f}\tARI-Macro: {:.4f}\tACTC-Macro: {:.4f}\tARTC-Macro: {:.4f}'.format(
            res_dict["Total-Macro"], res_dict["ARI-Macro"], res_dict["ACTC-Macro"], res_dict["ARTC-Macro"])
    return ARI_msg, ACTC_msg, ACTC_msg2, ARTC_msg, ARTC_msg2, macro_msg

def get_cdcp_eval_result2(res_dict):
    ARI_msg = 'ARI-Macro: {:.4f}\tRel: {:.4f}\tNo-Rel: {:.4f}'.format(
            res_dict["ARI-Macro"], res_dict["Rel"], res_dict["No-Rel"])
    ACTC_msg = 'ACTC-Macro: {:.4f}\tvalue: {:.4f}\tpolicy: {:.4f}\ttestimony: {:.4f}\tfact: {:.4f}\treference: {:.4f}'.format(
            res_dict["ACTC-Macro"], res_dict["value"], res_dict["policy"], res_dict["testimony"], res_dict["fact"], res_dict["reference"])
    ACTC_msg2 = 'ACTC-F1: {:.4f}'.format(res_dict["ACTC-F1"])
    ARTC_msg = 'ARTC-Macro: {:.4f}\treason: {:.4f}\tevidence: {:.4f}'.format(
        res_dict["ARTC-Macro"], res_dict["reason"], res_dict["evidence"])
    ARTC_msg2 = 'ARTC-F1: {:.4f}\tPre: {:.4f}\tRec: {:.4f}'.format(
        res_dict["ARTC-F1"], res_dict["Pre"], res_dict["Rec"])
    macro_msg = 'Total-Macro: {:.4f}\tARI-Macro: {:.4f}\tACTC-Macro: {:.4f}\tARTC-Macro: {:.4f}'.format(
            res_dict["Total-Macro"], res_dict["ARI-Macro"], res_dict["ACTC-Macro"], res_dict["ARTC-Macro"])
    macro_msg2 = 'Total-Macro-g: {:.4f}\tARI-Macro: {:.4f}\tACTC-Macro: {:.4f}\tARTC-Macro-g: {:.4f}'.format(
            res_dict["Total-Macro-g"], res_dict["ARI-Macro"], res_dict["ACTC-Macro"], res_dict["ARTC-Macro-g"])
    ARTC_msg3 = 'ARTC-Macro-g: {:.4f}\treason-g: {:.4f}\tevidence-g: {:.4f}'.format(
        res_dict["ARTC-Macro-g"], res_dict["reason-g"], res_dict["evidence-g"])
    return ARI_msg, ACTC_msg, ACTC_msg2, ARTC_msg, ARTC_msg2, macro_msg, macro_msg2, ARTC_msg3


def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return torch.from_numpy(adj_normalized.A).float()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    # print(f"sparse adj: {adj}")
    rowsum = np.array(adj.sum(1))
    # print(f"rowsum: {rowsum.shape}")
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # print(d_mat_inv_sqrt)
    #  D^(-1/2)AD^(-1/2)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).tocoo()

def chop_large_dis(tmp1_ari_group_label, span_num, ari_group_label):
    all_pair_num = (span_num*span_num - span_num) // 2
    group_size = len(tmp1_ari_group_label) // all_pair_num
    assert len(tmp1_ari_group_label) % all_pair_num == 0
    
    ari_pair_label = []
    for i in range(group_size):
        ari_group_label_i = ari_group_label[i]
        ari_pair_label_i = []
        for f in range(span_num):
            for t in range(f+1, span_num):
                if (f, t) in pair2sequence[span_num] and ((f, t) in ari_group_label_i or (t, f) in ari_group_label_i):
                    ari_pair_label_i.append((f, t))
        ari_pair_label.append(ari_pair_label_i)
    
    if span_num < 14:
        return tmp1_ari_group_label, ari_pair_label
    else:
        
        tmp2_ari_group_label = []
        for i in range(group_size):
            tmp1_ari_group_label_i = tmp1_ari_group_label[i*all_pair_num: i*all_pair_num+all_pair_num]
            
            matrix = np.zeros((span_num, span_num))
            f, t = np.triu_indices(span_num, k=1)
            matrix[f, t] = np.fromiter(tmp1_ari_group_label_i, dtype=np.int)
            fi = np.fromiter([f for f, _ in pair2sequence[span_num]], dtype=np.int)
            ti = np.fromiter([t for _, t in pair2sequence[span_num]], dtype=np.int)
            tmp2_ari_group_label_i = matrix[fi, ti]
            

            tmp2_ari_group_label += tmp2_ari_group_label_i.tolist()

        return tmp2_ari_group_label, ari_pair_label


    # elif span_num == 11:
    #     group_size = len(tmp1_ari_group_label) // 55
    #     assert len(tmp1_ari_group_label) % 55 == 0
    #     tmp2_ari_group_label = []
    #     for i in range(group_size):
    #         tmp1_ari_group_label_i = tmp1_ari_group_label[i: i+55]
    #         tmp2_ari_group_label_i = tmp1_ari_group_label_i[:9] + tmp1_ari_group_label_i[10:]
    #         tmp2_ari_group_label += tmp2_ari_group_label_i
    #     return tmp2_ari_group_label
    # elif span_num == 12:
    #     group_size = len(tmp1_ari_group_label) // 66
    #     assert len(tmp1_ari_group_label) % 66 == 0
    #     tmp2_ari_group_label = []
    #     for i in range(group_size):
    #         tmp1_ari_group_label_i = tmp1_ari_group_label[i: i+66]
    #         tmp2_ari_group_label_i = tmp1_ari_group_label_i[:9] + tmp1_ari_group_label_i[11:20] + tmp1_ari_group_label_i[21:]
    #         tmp2_ari_group_label += tmp2_ari_group_label_i
    #     return tmp2_ari_group_label
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = torch.Tensor([gamma])
        self.size_average = size_average
        if isinstance(alpha, (float, int, long)):
            if self.alpha > 1:
                raise ValueError('Not supported value, alpha should be small than 1.0')
            else:
                self.alpha = torch.Tensor([alpha, 1.0 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.alpha /= torch.sum(self.alpha)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # [N,C,H,W]->[N,C,H*W] ([N,C,D,H,W]->[N,C,D*H*W])
        # target
        # [N,1,D,H,W] ->[N*D*H*W,1]
        if self.alpha.device != input.device:
            self.alpha = torch.tensor(self.alpha, device=input.device)
        target = target.view(-1, 1)
        logpt = torch.log(input + 1e-10)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1, 1)
        pt = torch.exp(logpt)
        alpha = self.alpha.gather(0, target.view(-1))

        gamma = self.gamma

        if not self.gamma.device == input.device:
            gamma = torch.tensor(self.gamma, device=input.device)

        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss