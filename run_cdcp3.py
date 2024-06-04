import os, time, random, sys, json
import numpy as np
import logging
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from models.promptmodel_bart3_cdcp import E2EModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam, Adamax
import datetime
from dataloaders.dataloader_graph_bart_cdcp import ArgMiningDataset, generate_batch_fn, InfiniteIterator
from torch.utils.data import DataLoader
import pandas as pd
from configs.config import shared_configs
from tqdm import tqdm
from utils.misc import set_random_seed, NoOp, zero_none_grad
from utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
import math
from os.path import join
from utils.load_save import (ModelSaver, save_training_meta, load_state_dict_with_mismatch, E2E_TrainingRestorer)
from utils.basic_utils_cdcp import load_json, get_index_positions, flat_list_of_lists, get_edge_frompairs, \
	get_cdcp_eval_result, _pair2sequence, _sequence2pair, get_tuple_frompairs, args_metric, eval_edge_cdcp
from torch.nn.utils import clip_grad_norm_


# os.environ['CUDA_VISIBLE_DEVICES'] = config.device


def setup_model(cfg):
	LOGGER.info('Initializing model...')
	model = E2EModel(cfg)

	if cfg.e2e_weights_path:
		LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
		load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
	# else:
	# 	LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
	# 	LOGGER.info(f"Loading udg weights from {cfg.udg_weights_path}")
	# 	model.load_separate_ckpt(
	# 		bert_weights_path=cfg.bert_weights_path,
	# 		udg_weights_path=cfg.udg_weights_path
	# 	)

	if cfg.freeze_plm:
		model.freeze_plm_backbone()
	model.cuda()
	LOGGER.info('Model initialized.')

	for n, p in model.named_parameters():
		print(n, p.size())

	return model


def setup_dataloaders(config):
	LOGGER.info('Loading data...')

	data_df = pd.read_csv(config.data_path)
	with open(config.split_test_file_path, "r") as fp:
		test_id_list = json.load(fp)
	test_data_df = data_df[data_df["para_id"].isin(test_id_list)]
	train_data_df = data_df[~(data_df["para_id"].isin(test_id_list))]

	essay_id2parag_id_dict = train_data_df.groupby("essay_id").groups
	essay_id_list = list(essay_id2parag_id_dict.keys())
	random.shuffle(essay_id_list)
	num_train_essay = int(len(essay_id_list) * 0.9)
	dev_essay_id_list = essay_id_list[num_train_essay:]
	dev_para_id_list = []
	for essay_id in dev_essay_id_list:
		dev_para_id_list += essay_id2parag_id_dict[essay_id].tolist()

	dev_data_df = train_data_df[train_data_df["para_id"].isin(dev_para_id_list)]
	train_data_df = train_data_df[~train_data_df["para_id"].isin(dev_para_id_list)]

	train_data_df = train_data_df[train_data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]
	dev_data_df = dev_data_df[dev_data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]
	test_data_df = test_data_df[test_data_df["adu_spans"].apply(lambda x: len(eval(x)) > 0)]

	print(len(train_data_df), len(dev_data_df), len(test_data_df))

	train_dataset = ArgMiningDataset(train_data_df, config)
	dev_dataset = ArgMiningDataset(dev_data_df, config)
	test_dataset = ArgMiningDataset(test_data_df, config)

	train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
	                          shuffle=True, collate_fn=generate_batch_fn)
	dev_loader = DataLoader(dev_dataset, batch_size=config.val_batch_size,
	                        shuffle=False, collate_fn=generate_batch_fn)
	test_loader = DataLoader(test_dataset, batch_size=config.val_batch_size,
	                         shuffle=False, collate_fn=generate_batch_fn)
	LOGGER.info('Data loaded.')

	return train_loader, dev_loader, test_loader, len(train_dataset)


def build_optimizer_w_lr_mul(model_param_optimizer, learning_rate, weight_decay):
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	# Prepare optimizer
	param_optimizer = model_param_optimizer

	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer
		            if not any(nd in n for nd in no_decay)],
		 'weight_decay': weight_decay,
		 'lr': learning_rate},
		{'params': [p for n, p in param_optimizer
		            if any(nd in n for nd in no_decay)],
		 'weight_decay': 0.0,
		 'lr': learning_rate}]
	return optimizer_grouped_parameters


def setup_optimizer(model, opts):
	"""model_type: str, one of [transformer, cnn]"""

	plm_param_optimizer = [
		(n, p) for n, p in list(model.named_parameters())
		if "rel_g" not in n and p.requires_grad] # update the rel_gat to rel_g after 20230602 20:17
	prompt_param_optimizer = [
		(n, p) for n, p in list(model.named_parameters())
		if "rel_g" in n and p.requires_grad]

	plm_grouped_parameters = build_optimizer_w_lr_mul(
		plm_param_optimizer, opts.plm_learning_rate, opts.plm_weight_decay)
	prompt_grouped_parameters = build_optimizer_w_lr_mul(
		prompt_param_optimizer, opts.learning_rate, opts.weight_decay)

	optimizer_grouped_parameters = []
	optimizer_grouped_parameters.extend(plm_grouped_parameters)
	optimizer_grouped_parameters.extend(prompt_grouped_parameters)
	if opts.optim == 'adam':
		OptimCls = Adam
	elif opts.optim == 'adamax':
		OptimCls = Adamax
	elif opts.optim == 'adamw':
		OptimCls = AdamW
	else:
		raise ValueError('invalid optimizer')
	optimizer = OptimCls(optimizer_grouped_parameters, lr=opts.learning_rate, betas=opts.betas)
	return optimizer

best_val_macro = 0
best_v2macro = 0
flag = 0

best_macro = 0


# @torch.no_grad()
def validate(model, val_loader, cfg, train_global_step, mode='dev', saver=None):
	"""use eval_score=False when doing inference on test sets where answers are not available"""
	LOGGER.info('*' * 20 + f"The performance on {mode} set" + '*' * 20)

	model.eval()
	st = time.time()
	debug_step = 5
	global best_macro, best_val_macro, best_v2macro, flag

	all_pred_actc = []
	all_label_actc = []
	all_pred_ari = []
	all_label_ari = []
	# all_pred_ari2 = []
	# all_label_ari2 = []
	all_pred_artc = []
	all_label_artc = []
	all_pred_artc2 = []
	all_label_artc2 = []
	# 'value': 2158, 'policy': 810, 'testimony': 1026, 'fact': 746, 'reference': 32
	eval_res = {"ARI-Macro": None, "Rel": None, "No-Rel": None,
	            "ARTC-Macro": None, "reason": None, "evidence": None,
	            "ARTC2-Macro": None, "F1": None, "Pre": None, "Rec": None,
	            "ACTC-Macro": None, "value": None, "policy": None, "testimony": None, "fact": None, "reference": None,
	            "ACTC-F1": None,
	            "Total-Macro": None}

	for val_step, batch in enumerate(val_loader):
		# forward pass
		para_tokens_ids_list = batch['para_tokens_ids']
		# AC_bow_lists = batch['bow_vecs']
		AC_spans_list = batch['AC_spans']
		whole_graph_list = batch['whole_graph']
		# AC_positions_list, AC_para_types_list = batch['AC_positions'], batch['AC_para_types']
		true_AC_types_list, true_AR_pairs_list = batch['true_AC_types'], batch['true_AR_pairs']
		true_AR_link_type_list = batch['true_AR_link_types']

		# flat_true_AC_types_list = [_ for AC_types in true_AC_types_list for _ in AC_types]
		# true_AC_types_tensor = torch.tensor(flat_true_AC_types_list).cuda()
		#
		# if true_AC_types_tensor.shape[0] <= 1:
		#    continue

		# if cfg.debug and val_step >= debug_step:
		#    break

		actc_logits_list, ari_logits_list, artc_logits_list = model(
			para_tokens_ids_list,
			# AC_bow_lists,
			AC_spans_list,
			whole_graph_list,
			# AC_positions_list,
			# AC_para_types_list,
			mode='val')

		# print("para_tokens_ids_list", para_tokens_ids_list)
		# print("AC_spans_list", AC_spans_list)
		# print("para_srl", para_srl)

		span_num_list = [len(para_AC_spans) for para_AC_spans in AC_spans_list]
		span_num_set = set(span_num_list)

		for idx, span_num in enumerate(span_num_set):
			para_index = get_index_positions(span_num_list, span_num)
			group_size = len(para_index)

			if span_num == 1:
				actc_group_label = list(map(true_AC_types_list.__getitem__, para_index))
				actc_group_label_list = flat_list_of_lists(actc_group_label)
				all_label_actc += actc_group_label_list
				actc_group_pred = torch.argmax(actc_logits_list[idx], dim=-1)  # [group_size * span_num, class_num]
				all_pred_actc += actc_group_pred.tolist()
				# print("AC_types_group_label_list", AC_types_group_label_list)
				# print("pred_AC_types_list",pred_AC_types_list)

			else:
				actc_group_label = list(map(true_AC_types_list.__getitem__, para_index))
				actc_group_label_list = flat_list_of_lists(actc_group_label)
				all_label_actc += actc_group_label_list
				actc_group_pred = torch.argmax(actc_logits_list[idx] , dim=-1) # [group_size * span_num, class_num]
				all_pred_actc +=  actc_group_pred.tolist()
				# print("AC_types_group_label_list", AC_types_group_label_list)
				# print("pred_AC_types_list",pred_AC_types_list)

				ari_group_label = list(map(true_AR_pairs_list.__getitem__, para_index))
				tmp1_ari_group_label = get_edge_frompairs(ari_group_label, span_num) # [group_size * all_pair_num]
				all_label_ari +=  tmp1_ari_group_label
				# tmp2_ari_group_label = _pair2sequence(ari_group_label, span_num) # [group_size, dis_pair_num]
				# all_label_ari2 += tmp2_ari_group_label

				ari_group_pred = torch.argmax(ari_logits_list[idx], dim=-1)  # [group_size * pair_num, class_num]
				# print("ari_logits_list[idx]", ari_logits_list[idx].size(), idx, span_num)
				ari_triu_pred, ari_pair_pred = _sequence2pair(ari_group_pred, span_num, group_size) #
				all_pred_ari += ari_triu_pred
				# all_pred_ari2 += ari_group_pred.int().tolist()

				artc_group_label = list(map(true_AR_link_type_list.__getitem__, para_index))

				all_label_artc2 += get_tuple_frompairs(ari_group_label, artc_group_label)
				artc_group_pred = torch.argmax(artc_logits_list[idx], dim=-1) # [group_size * pair_num, class_num]
				# print("artc_group_pred", artc_group_pred.size(), len(artc_group_label), span_num)
				artc_group_pred_list = torch.masked_select(artc_group_pred, ari_group_pred.bool()).split([len(l) for l in ari_pair_pred])
				all_pred_artc2 += get_tuple_frompairs(ari_pair_pred, artc_group_pred_list)

				tmp1_artc_group_label = get_edge_frompairs(ari_group_label, span_num, artc_group_label, value=2)  # [group_size * all_pair_num]
				all_label_artc += tmp1_artc_group_label
				# artc_triu_pred, _ = _sequence2pair(artc_group_pred, span_num, group_size, value=2)
				artc_triu_pred = get_edge_frompairs(ari_pair_pred, span_num, artc_group_pred.view(group_size, -1).tolist(), value=2)
				all_pred_artc += artc_triu_pred

				# print("ari_group_label", ari_group_label)
				# print("ari_pair_pred", ari_pair_pred)
				# print("artc_group_label", artc_group_label, span_num)
				# print("tmp1_artc_group_label", tmp1_artc_group_label)
				# print("artc_triu_pred", artc_triu_pred)
				# print("label_artc2", get_tuple_frompairs(ari_group_label, artc_group_label))
				# print("pred_artc2", get_tuple_frompairs(ari_pair_pred, artc_group_pred_list))
				# print("*" * 25)

	ari_metric = f1_score(all_label_ari, all_pred_ari, labels=[0, 1], average=None)
	eval_res["No-Rel"], eval_res["Rel"] = ari_metric
	eval_res["ARI-Macro"] = (eval_res["Rel"] + eval_res["No-Rel"]) / 2

	actc_metric = f1_score(all_label_actc, all_pred_actc, labels=[0, 1, 2, 3, 4], average=None)
	eval_res["value"], eval_res["policy"], eval_res["testimony"], eval_res["fact"], eval_res["reference"] = actc_metric
	eval_res['ACTC-Macro'] = (eval_res["value"] + eval_res["policy"] + eval_res["testimony"] + eval_res["fact"] + eval_res["reference"]) / 5

	actc2_dic = f1_score(all_label_actc, all_pred_actc, labels=[0, 1, 2, 3, 4], average="micro")
	eval_res["ACTC-F1"] = actc2_dic

	# artc_metric = f1_score(all_label_artc, all_pred_artc, labels=[0, 1], average=None)
	# eval_res["Sup"], eval_res["Atc"] = artc_metric
	# eval_res["ARTC-Macro"] = (eval_res["Sup"] + eval_res["Atc"]) / 2

	edge_dic = eval_edge_cdcp(all_pred_artc2, all_label_artc2)
	eval_res["reason"], eval_res["evidence"] = edge_dic["reason"]['f'], edge_dic["evidence"]['f']
	eval_res["ARTC-Macro"] = (eval_res["reason"] + eval_res["evidence"]) / 2

	artc2_dic = args_metric(all_label_artc2, all_pred_artc2) # {'pre': pre, 'rec': rec, 'f1': f1, 'acc': acc}
	eval_res["ARTC-F1"], eval_res["Pre"], eval_res["Rec"] = artc2_dic['f1'], artc2_dic['pre'], artc2_dic['rec']

	eval_res["Total-Macro"] = (eval_res["ARI-Macro"] + eval_res["ACTC-Macro"] + eval_res["ARTC-Macro"]) / 3

	if mode == "test" and eval_res["Total-Macro"] > best_macro:
		best_macro = eval_res["Total-Macro"]
		# saver.save(0, model)

	if mode == "dev" and eval_res["Total-Macro"] > best_val_macro:
		best_val_macro = eval_res["Total-Macro"]
		flag = 1
	if mode == "test" and flag == 1:
		best_v2macro = eval_res["Total-Macro"]
		flag = 0


	# if eval_res["Total-Macro"] > best_f1:
	# early_stop = 0
	# best_f1 = eval_res["Total-Macro"]
	best_global_step = train_global_step

	ARI_msg, ACTC_msg, ACTC_msg2, ARTC_msg, ARTC_msg2, macro_msg = get_cdcp_eval_result(eval_res)
	LOGGER.info(ARI_msg)
	LOGGER.info(ACTC_msg)
	LOGGER.info(ACTC_msg2)
	LOGGER.info(ARTC_msg)
	LOGGER.info(ARTC_msg2)
	LOGGER.info(macro_msg)

	LOGGER.info('BEST Val Macro: {:.4f}'.format(best_val_macro))
	LOGGER.info('BEST Test Macro: {:.4f}'.format(best_v2macro))
	LOGGER.info(f"{mode} finished in {int(time.time() - st)} seconds.")
	model.train()


def start_training(cfg):
	set_random_seed(cfg.seed)

	# prepare data
	train_loader, dev_loader, test_loader, total_n_examples = setup_dataloaders(cfg)

	# compute the number of steps and update cfg
	# total_n_examples = len(train_loader.dataset)
	total_train_batch_size = int(cfg.train_batch_size * cfg.gradient_accumulation_steps)
	cfg.num_train_steps = int(math.ceil(
		1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))
	cfg.num_warmup_steps = int(cfg.num_train_steps * cfg.warmup_ratio)

	cfg.valid_steps = int(math.ceil(
		1. * cfg.num_train_steps / cfg.num_valid /
		cfg.min_valid_steps)) * cfg.min_valid_steps
	actual_num_valid = int(math.floor(
		1. * cfg.num_train_steps / cfg.valid_steps)) + 1

	# setup model and optimizer
	model = setup_model(cfg)
	model.train()

	optimizer = setup_optimizer(model, cfg)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.num_warmup_steps,
	                                            num_training_steps=cfg.num_train_steps)

	# restore
	now_time = datetime.datetime.now()
	now_time = now_time.strftime("%Y-%m-%d-%X-%a")
	savePath = join(cfg.output_dir, now_time)
	cfg.output_dir = savePath  # if you need restore your checkpoint, please annotate here and use the correct outdir in order line

	# restore
	restorer = E2E_TrainingRestorer(cfg, model=model, optimizer=optimizer)
	global_step = restorer.global_step
	TB_LOGGER.global_step = global_step

	LOGGER.info("Saving training meta...")
	save_training_meta(cfg)
	LOGGER.info("Saving training done...")

	TB_LOGGER.create(join(cfg.output_dir, 'log'))
	pbar = tqdm(total=cfg.num_train_steps)
	model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
	add_log_to_file(join(cfg.output_dir, "log", "log.txt"))

	# torch.save(model.state_dict(), join(cfg.output_dir, "model_init.pt"))

	if global_step > 0:
		pbar.update(global_step)

	validate(model, dev_loader, cfg, global_step, 'dev')
	validate(model, test_loader, cfg, global_step, 'test', model_saver)

	LOGGER.info(cfg)
	LOGGER.info("Starting training...")
	LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
	LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
	LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
	            f"Accumulate steps = {total_train_batch_size}")
	LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
	LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
	LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")

	debug_step = 3
	running_loss = RunningMeter('train_loss')
	running_loss_ari = RunningMeter('train_loss_ari')
	running_loss_actc = RunningMeter('train_loss_actc')
	running_loss_artc = RunningMeter('train_loss_artc')
	running_loss_mlm = RunningMeter('train_loss_mlm')
	step = 0
	for epoch in range(cfg.num_train_epochs):
		LOGGER.info(f'Start training in epoch: {epoch}')
		# for batch in InfiniteIterator(train_loader):

		for batch in train_loader:

			n_epoch = int(1. * total_train_batch_size * global_step / total_n_examples)
			# LOGGER.info("Running epoch: {}".format(n_epoch))

			# forward pass
			para_tokens_ids_list = batch['para_tokens_ids']
			# AC_bow_lists = batch['bow_vecs']
			AC_spans_list = batch['AC_spans']
			whole_graph_list = batch['whole_graph']
			# AC_positions_list, AC_para_types_list = batch['AC_positions'], batch['AC_para_types']
			true_AC_types_list, true_AR_pairs_list = batch['true_AC_types'], batch['true_AR_pairs']
			true_AR_link_type_list = batch['true_AR_link_types']

			# flat_true_AC_types_list = [_ for AC_types in true_AC_types_list for _ in AC_types]
			# true_AC_types_tensor  = torch.tensor(flat_true_AC_types_list).cuda()
			#
			# if true_AC_types_tensor.shape[0] <= 1:
			#    continue
			if len(true_AC_types_list) == 1 and len(true_AC_types_list[0]) == 1:
				continue

			ml_loss, actc_loss, ari_loss, artc_loss = model(para_tokens_ids_list,
			                                             # AC_bow_lists,
			                                             AC_spans_list,
			                                             whole_graph_list,
			                                             # AC_positions_list,
			                                             # AC_para_types_list,
			                                             true_AC_types_list,
			                                             true_AR_pairs_list,
			                                             true_AR_link_type_list)

			# print("ml_loss", ml_loss.size())
			# print("actc_loss", actc_loss.size())
			# print("ari_loss", ari_loss.size())
			# print("artc_loss", artc_loss.size())

			loss = ml_loss + actc_loss + ari_loss + artc_loss

			if cfg.gradient_accumulation_steps > 1:
				loss = loss / cfg.gradient_accumulation_steps

			loss.backward()

			running_loss(loss.item())
			running_loss_ari(ari_loss.item())
			running_loss_artc(artc_loss.item())
			running_loss_actc(actc_loss.item())
			running_loss_mlm(ml_loss.item())
			# backward pass
			# optimizer
			if (step + 1) % cfg.gradient_accumulation_steps == 0:
				global_step += 1

				# learning rate scheduling

				TB_LOGGER.add_scalar('train/loss', running_loss.val, global_step)
				TB_LOGGER.add_scalar('train/loss_ari', running_loss_ari.val, global_step)
				TB_LOGGER.add_scalar('train/loss_actc', running_loss_actc.val, global_step)
				TB_LOGGER.add_scalar('train/loss_artc', running_loss_artc.val, global_step)
				TB_LOGGER.add_scalar('train/loss_mlm', running_loss_mlm.val, global_step)

				# update model params
				if cfg.grad_norm != -1:
					grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_norm)
					TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)

				# Check if there is None grad
				# none_grads = [
				#     p[0] for p in model.named_parameters()
				#     if p[1].requires_grad and p[1].grad is None]
				# print(len(none_grads), none_grads)
				# assert len(none_grads) == 2, f"{none_grads}"

				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()
				restorer.step()
				pbar.update(1)

				# print(len(optimizer.param_groups))
				assert len(optimizer.param_groups) == 4
				for pg_n, param_group in enumerate(optimizer.param_groups):
					if pg_n == 0:
						lr_this_step_transformer = param_group['lr']
					elif pg_n == 2:
						lr_this_step_udg = param_group['lr']

				TB_LOGGER.add_scalar(
					"train/lr_plm", lr_this_step_transformer, global_step)
				TB_LOGGER.add_scalar(
					"train/lr_graph", lr_this_step_udg, global_step)

				TB_LOGGER.step()

				# checkpoint
				if global_step % cfg.valid_steps == 0:
					LOGGER.info(f'Step {global_step}: start validation in epoch: {n_epoch}')
					validate(model, dev_loader, cfg, global_step, 'dev')
					validate(model, test_loader, cfg, global_step, 'test', model_saver)
					# model_saver.save(step=global_step, model=model)
			if global_step >= cfg.num_train_steps:
				break

			if cfg.debug and global_step >= debug_step:
				break

			step += 1

	if global_step % cfg.valid_steps != 0:
		LOGGER.info(f'Step {global_step}: start validation in epoch: {n_epoch}')
		validate(model, dev_loader, cfg, global_step, 'dev')
		validate(model, test_loader, cfg, global_step, 'test', model_saver)
		# model_saver.save(step=global_step, model=model)


if __name__ == '__main__':
	cfg = shared_configs.get_cdcp_args()
	start_training(cfg)
