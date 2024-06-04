from models.prompt_bart7_1 import JointPrompt
from torch import nn
from utils.load_save import load_state_dict_with_mismatch
from utils.basic_utils import get_index_positions
import torch
import numpy as np
import torch.nn.functional as F
from models.pos_map import pair_idx_map, pair2sequence, bart_prefix_ac_map7, pair2sequence


class E2EModel(nn.Module):
	def __init__(self, config):
		super(E2EModel, self).__init__()
		self.config = config
		# self.tokenizer = BertTokenizer.from_pretrained(config.bert_weights_path)
		self.jp_model = JointPrompt(config)

		self.num_class = 7
		self.max_ac_num = config.max_AC_num
		self.max_pair_num = 63 #
		self.pair_num_map = {1:0, 2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 54, 12:63}
		self.mask_type = config.mask_type

	def forward(self,
	            para_tokens_ids_list,
	            AC_spans_list,
	            whole_graph_list,
	            true_AC_types_list=None,
	            true_AR_pairs_list=None,
	            true_AR_link_types_list=None,
	            mode='train'):

		if mode == 'train':
			loss1, loss2, loss3, loss4 = self.train_mode(para_tokens_ids_list,
			                                      AC_spans_list,
			                                      whole_graph_list,
			                                      true_AC_types_list,
			                                      true_AR_pairs_list,
			                                      true_AR_link_types_list)
			result = (loss1, loss2, loss3, loss4)

		elif mode == 'val':
			list1, list2, list3 = self.val_mode(para_tokens_ids_list, AC_spans_list, whole_graph_list)
			result = (list1, list2, list3)
		else:
			raise ValueError("Error model mode , please choice [train] or [val]")

		return result

	def train_mode(self, para_tokens_ids_list,
	               AC_spans_list,
	               whole_graph_list,
	               true_AC_types_list=None,
	               true_AR_pairs_list=None,
	               true_AR_link_types_list=None):
		span_num_list = [len(para_AC_spans) for para_AC_spans in AC_spans_list]
		span_num_set = set(span_num_list)

		loss_mlm_list = []
		loss_actc_list = []
		loss_ari_list = []
		loss_artc_list = []

		all_pair_type_num = 0
		for span_num in span_num_set:

			para_index = get_index_positions(span_num_list, span_num)
			group_size = len(para_index)

			group_tokens_ids_list = list(map(para_tokens_ids_list.__getitem__, para_index))

			AC_group_spans = list(map(AC_spans_list.__getitem__, para_index))

			group_whole_graphs = list(map(whole_graph_list.__getitem__, para_index))

			true_group_AC_types = list(map(true_AC_types_list.__getitem__, para_index))
			true_group_AC_types_label = torch.LongTensor(true_group_AC_types).cuda().view(-1)

			if span_num > 1:
				true_AR_group_pairs = list(map(true_AR_pairs_list.__getitem__, para_index))
				true_AR_link_group_types = list(map(true_AR_link_types_list.__getitem__, para_index))

				true_group_AR_label, true_group_AR_type_label, pair_type_num = self._pair2sequence(true_AR_group_pairs, span_num, true_AR_link_group_types)
				all_pair_type_num += pair_type_num
			else:
				true_group_AR_label = None
				true_group_AR_type_label = None

			input_ids_list = []
			tokens_num_list = []
			prompt_ids_list = []
			for tokens_ids_list in group_tokens_ids_list:
				tokens_num_list.append(len(tokens_ids_list))
				input_ids_list.append(tokens_ids_list)
				prompt_ids_list.append(list(bart_prefix_ac_map7[span_num]))
			input_ids_list, mask_list = self.padding_and_mask_forpara(input_ids_list)
			prompt_ids = torch.LongTensor(prompt_ids_list).cuda()
			cross_mask = self.contruct_cross_mask(mask_list, AC_group_spans, tokens_num_list, span_num)
			group_output = self.jp_model(
				input_ids=torch.LongTensor(input_ids_list).cuda(),
				AC_spans_list=AC_group_spans,
				prompt_ids=prompt_ids,
				adjs=torch.Tensor(group_whole_graphs).cuda(),
				span_num=span_num,
				attention_mask=torch.Tensor(mask_list).cuda(),
				decoder_attention_mask=torch.ones_like(prompt_ids, device=prompt_ids.device),
				corss_attention_mask=cross_mask,
				labels=(true_group_AC_types_label, true_group_AR_label, true_group_AR_type_label)
			)

			if group_output['ml_loss'] != None:
				loss_mlm_list.append(group_output['ml_loss'])
			loss_actc_list.append(group_output['actc_loss'])
			if group_output['ari_loss'] != None:
				loss_ari_list.append(group_output['ari_loss'])
			if group_output['artc_loss'] != None:
				loss_artc_list.append(group_output['artc_loss'])

		all_pair_type_num = 1 if all_pair_type_num == 0 else all_pair_type_num
		if loss_mlm_list != []:
			ml_loss = torch.cat(loss_mlm_list).mean()
		else:
			ml_loss = torch.tensor(0).cuda()

		# TODO equally compute the three loss by summing them
		actc_loss = torch.cat(loss_actc_list).mean()
		if loss_ari_list != []:
			ari_loss = torch.cat(loss_ari_list).mean()
		else:
			ari_loss = torch.tensor(0).cuda()
		if loss_artc_list != []:
			artc_loss = torch.cat(loss_artc_list).sum() / all_pair_type_num
		else:
			artc_loss = torch.tensor(0).cuda()

		return ml_loss, actc_loss, ari_loss, artc_loss

	def padding_and_mask_forpara(self, ids_list):
		max_len = max([len(x) for x in ids_list])
		mask_list = []
		ids_padding_list = []
		for ids in ids_list:
			mask = [1] * len(ids) + [0] * (max_len - len(ids))
			ids = ids + [0] * (max_len - len(ids))
			mask_list.append(mask)
			ids_padding_list.append(ids)
		return ids_padding_list, mask_list

	def contruct_cross_mask(self, mask_list, span_list, tokens_num_list, span_num):
		if self.mask_type == "full":
			masks = torch.Tensor(mask_list).cuda()
		elif self.mask_type == "global_sys":
			masks = np.zeros((len(mask_list), len(mask_list[0]), len(mask_list[0])))
			for i in range(len(span_list)):
				masks_i = np.zeros((sum(mask_list[i]), sum(mask_list[i])))
				masks_i[:tokens_num_list[i], :tokens_num_list[i]] = 1 #np.ones((tokens_num_list[i], tokens_num_list[i]))
				masks_i[-1:, :] = 1 #
				masks_i[:, -1:] = 1
				masks_i[tokens_num_list[i]: tokens_num_list[i] + 1, :] = 1 # doc token can see all tokens (including text and prompt)
				masks_i[:, tokens_num_list[i]: tokens_num_list[i] + 1] = 1 # all tokens (including text and prompt) tokens can see doc token
				for j, span in enumerate(span_list[i]):
					masks_i[tokens_num_list[i] + 1 + 2 * j:tokens_num_list[i] + 1 + 2 + 2 * j,
					tokens_num_list[i] + 1 + 2 * j:tokens_num_list[i] + 1 + 2 + 2 * j] = 1 # prompt + label tokens can see itself
					masks_i[span[0]: span[1]+1, tokens_num_list[i]+1+2*j:tokens_num_list[i]+1+2+2*j] = 1 # prompt + label tokens can see its AC tokens
					masks_i[tokens_num_list[i] + 1 + 2 * j:tokens_num_list[i] + 1 + 2 + 2 * j,
					span[0]: span[1] + 1] = 1 # AC tokens can see its prompt + label tokens

				if span_num >= 2:
					actc_p_num = 2 * span_num
					ari_p_num = 2 * self.pair_num_map[span_num]
					spans = span_list[i]
					for j, (a1, a2) in enumerate(list(pair2sequence[span_num])):
						masks_i[
						tokens_num_list[i] + 1 + actc_p_num + 2 * j:tokens_num_list[i] + 1 + actc_p_num + 2 + 2 * j,
						tokens_num_list[i] + 1 + actc_p_num + 2 * j:tokens_num_list[
							                                            i] + 1 + actc_p_num + 2 + 2 * j] = 1  # ari pair prompt + label tokens can see itself
						masks_i[
						tokens_num_list[i] + 1 + actc_p_num + ari_p_num + 2 * j:tokens_num_list[
							                                                        i] + 1 + actc_p_num + ari_p_num + 2 + 2 * j,
						tokens_num_list[i] + 1 + actc_p_num + ari_p_num + 2 * j:tokens_num_list[
							                                                        i] + 1 + actc_p_num + ari_p_num + 2 + 2 * j] = 1  # artc pair prompt + label tokens can see itself
						masks_i[
						tokens_num_list[i] + 1 + actc_p_num + 2 * j:tokens_num_list[i] + 1 + actc_p_num + 2 + 2 * j,
						spans[a1][0]:spans[a1][1] + 1] = 1  # ari prompt + label tokens can see its AC pair
						masks_i[
						tokens_num_list[i] + 1 + actc_p_num + 2 * j:tokens_num_list[i] + 1 + actc_p_num + 2 + 2 * j,
						spans[a2][0]:spans[a2][1] + 1] = 1  # ari prompt + label tokens can see its AC pair
						masks_i[spans[a1][0]:spans[a1][1] + 1,
						tokens_num_list[i] + 1 + actc_p_num + 2 * j:tokens_num_list[
							                                            i] + 1 + actc_p_num + 2 + 2 * j] = 1  # AC pair can see its ari prompt + label tokens
						masks_i[spans[a2][0]:spans[a2][1] + 1,
						tokens_num_list[i] + 1 + actc_p_num + 2 * j:tokens_num_list[
							                                            i] + 1 + actc_p_num + 2 + 2 * j] = 1  # AC pair can see its ari prompt + label tokens

						masks_i[
						tokens_num_list[i] + 1 + actc_p_num + ari_p_num + 2 * j:tokens_num_list[
							                                                        i] + 1 + actc_p_num + ari_p_num + 2 + 2 * j,
						spans[a1][0]:spans[a1][1] + 1] = 1  # artc prompt + label tokens can see its AC pair
						masks_i[
						tokens_num_list[i] + 1 + actc_p_num + ari_p_num + 2 * j:tokens_num_list[
							                                                        i] + 1 + actc_p_num + ari_p_num + 2 + 2 * j,
						spans[a2][0]:spans[a2][1] + 1] = 1  # artc prompt + label tokens can see its AC pair
						masks_i[spans[a1][0]:spans[a1][1] + 1,
						tokens_num_list[i] + 1 + actc_p_num + ari_p_num + 2 * j:tokens_num_list[
							                                                        i] + 1 + actc_p_num + ari_p_num + 2 + 2 * j] = 1  # AC pair can see its artc prompt + label tokens
						masks_i[spans[a2][0]:spans[a2][1] + 1,
						tokens_num_list[i] + 1 + actc_p_num + ari_p_num + 2 * j:tokens_num_list[
							                                                        i] + 1 + actc_p_num + ari_p_num + 2 + 2 * j] = 1  # AC pair can see its artc prompt + label tokens


				masks[i, :sum(mask_list[i]), :sum(mask_list[i])] = masks_i
			masks = torch.Tensor(masks).cuda()

		else:
			raise ValueError
		return masks

	def padding_and_mask_forpara2(self, ids_list, span_list, tokens_num_list):
		max_len = max([len(x) for x in ids_list])
		if self.mask_type == "full":
			mask_list = []
			ids_padding_list = []
			for ids in ids_list:
				mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
				ids = ids + [0] * (max_len - len(ids))
				mask_list.append(mask)
				ids_padding_list.append(ids)
			input_ids = torch.LongTensor(ids_padding_list).cuda()
			masks = torch.tensor(mask_list).cuda()
		elif self.mask_type == "global_sys":
			masks = np.zeros((len(ids_list), max_len, max_len))
			mask_list = []
			ids_padding_list = []
			for ids in ids_list:
				mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
				ids = ids + [0] * (max_len - len(ids))
				mask_list.append(mask)
				ids_padding_list.append(ids)

			for i in range(len(span_list)):
				masks_i = np.zeros((sum(mask_list[i]), sum(mask_list[i])))
				masks_i[:tokens_num_list[i], :tokens_num_list[i]] = 1 #np.ones((tokens_num_list[i], tokens_num_list[i]))
				masks_i[-1:, :] = 1 #
				masks_i[:, -1:] = 1
				masks_i[tokens_num_list[i]: tokens_num_list[i]+1, :] = 1 # doc token can see all tokens (including text and prompt)
				masks_i[:, tokens_num_list[i]: tokens_num_list[i] + 1] = 1 # all tokens (including text and prompt) tokens can see doc token
				for j, span in span_list[i]:
					masks_i[tokens_num_list[i] + 1 + 1 + 2 * j:tokens_num_list[i] + 1 + 3 + 2 * j,
					tokens_num_list[i] + 1 + 1 + 2 * j:tokens_num_list[i] + 1 + 3 + 2 * j] = 1 # prompt + label tokens can see itself
					masks_i[span[0]: span[1]+1, tokens_num_list[i]+1+1+2*j:tokens_num_list[i]+1+3+2*j] = 1 # prompt + label tokens can see its AC tokens
					masks_i[tokens_num_list[i] + 1 + 1 + 2 * j:tokens_num_list[i] + 1 + 3 + 2 * j,
					span[0]: span[1] + 1] = 1 # AC tokens can see its prompt + label tokens
				masks[i, :sum(mask_list[i]), :sum(mask_list[i])] = masks_i
			masks = torch.tensor(masks).cuda()

		return input_ids, masks

	def _pair2sequence(self, pair_list, span_num, pair_type_list):
			pairs_true = []
			pair_type_true = []
			pairs_num = len(pair_idx_map[span_num])
			pair_type_num = 0
			for i in range(len(pair_list)):
				true_indices = []
				tmp_pair_types = []
				for j, x in enumerate(pair_list[i]):
					if max(x) - min(x) <= 9:
						true_indices.append(list(pair2sequence[span_num]).index((min(x), max(x))))
						tmp_pair_types.append(pair_type_list[i][j])

				# true_indices = [list(pair2sequence[span_num]).index((min(x), max(x))) for x in pair_list[i] if max(x) - min(x) <= 9]
				temp = torch.zeros(pairs_num)
				temp[true_indices] = 1.
				pairs_true.append(temp)

				temp_type = torch.ones(pairs_num) * -100
				temp_type[true_indices] = torch.Tensor(tmp_pair_types) # pair_type_list[i][:-2] if len(pair_type_list[i]) == 11 else pair_type_list[i]
				pair_type_true.append(temp_type)
				pair_type_num += len(tmp_pair_types) # pair_type_list[i]
			pairs_true = torch.cat(pairs_true).long().cuda()
			pair_type_true = torch.cat(pair_type_true).long().cuda()
			return pairs_true, pair_type_true, pair_type_num

	def val_mode(self, para_tokens_ids_list, AC_spans_list, whole_graph_list):
		span_num_list = [len(para_AC_spans) for para_AC_spans in AC_spans_list]
		span_num_set = set(span_num_list)

		actc_logits_list = []
		ari_logits_list = []
		artc_logits_list = []

		for span_num in span_num_set:
			para_index = get_index_positions(span_num_list, span_num)
			group_size = len(para_index)
			group_tokens_ids_list = list(map(para_tokens_ids_list.__getitem__, para_index))

			AC_group_spans = list(map(AC_spans_list.__getitem__, para_index))
			group_whole_graphs = list(map(whole_graph_list.__getitem__, para_index))

			input_ids_list = []
			tokens_num_list = []
			prompt_ids_list = []
			for tokens_ids_list in group_tokens_ids_list:
				tokens_num_list.append(len(tokens_ids_list))
				input_ids_list.append(tokens_ids_list)
				prompt_ids_list.append(list(bart_prefix_ac_map7[span_num]))
			input_ids_list, mask_list = self.padding_and_mask_forpara(input_ids_list)
			prompt_ids = torch.LongTensor(prompt_ids_list).cuda()
			cross_mask = self.contruct_cross_mask(mask_list, AC_group_spans, tokens_num_list, span_num)
			actc_scores, ari_scores, artc_scores = self.jp_model.predict(
				input_ids=torch.LongTensor(input_ids_list).cuda(),
				AC_spans_list=AC_group_spans,
				prompt_ids=prompt_ids,
				adjs=torch.Tensor(group_whole_graphs).cuda(),
				span_num=span_num,
				attention_mask=torch.Tensor(mask_list).cuda(),
				decoder_attention_mask=torch.ones_like(prompt_ids, device=prompt_ids.device),
				corss_attention_mask=cross_mask
			)

			actc_logits_list.append(actc_scores)
			ari_logits_list.append(ari_scores)
			artc_logits_list.append(artc_scores)

		return actc_logits_list, ari_logits_list, artc_logits_list

	def freeze_plm_backbone(self):
		for n, p in self.bert_encoder.named_parameters():
			if 'embeding' not in n:
				p.requires_grad = False
