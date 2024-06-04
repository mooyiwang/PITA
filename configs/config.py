"""
Modified from UNITER code
"""
import os
import sys
import json
import argparse

from easydict import EasyDict as edict


def parse_with_config(parsed_args):
    """This function will set args based on the input config file.
    (1) it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    (2) it also sets configs in the config file but declared in the parser
    """
    # convert to EasyDict object, enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


def str2bool(v):
    if isinstance(v, bool) and v == True:
        return True
    if isinstance(v, bool) and v == False:
        return False
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class SharedConfigs(object):
    """Shared options for pre-training and downstream tasks.
    For each downstream task, implement a get_*_args function,
    see `get_pretraining_args()`

    Usage:
    >>> shared_configs = SharedConfigs()
    >>> pretraining_config = shared_configs.get_pretraining_args()
    """

    def __init__(self, desc="shared config for pretraining and finetuning"):
        parser = argparse.ArgumentParser(description=desc)
        # debug parameters
        parser.add_argument(
            "--debug", type=int, choices=[0, 1], default=0,
            help="debug mode, output extra info & break all loops."
                 "0: disable, 1 enable")
        parser.add_argument(
            "--data_ratio", type=float, default=1.0,
            help="portion of train/val exampels to use,"
                 "e.g., overfit a small set of data")

        # Required parameters
        parser.add_argument(
            "--model_config", type=str,
            help="path to model structure config json")
        parser.add_argument(
            "--tokenizer_dir", type=str, help="path to tokenizer dir")
        parser.add_argument(
            "--output_dir", type=str,
            help="dir to store model checkpoints & training meta.")

        # data preprocessing parameters
        parser.add_argument(
            "--max_txt_len", type=int, default=20, help="max text #tokens ")
        
        # training parameters
        parser.add_argument(
            "--train_batch_size", default=32, type=int,
            help="Single-GPU batch size for training for Horovod.")
        parser.add_argument(
            "--val_batch_size", default=128, type=int,
            help="Single-GPU batch size for validation for Horovod.")
        parser.add_argument(
            "--gradient_accumulation_steps", type=int, default=1,
            help="#updates steps to accumulate before performing a backward/update pass."
                 "Used to simulate larger batch size training. The simulated batch size "
                 "is train_batch_size * gradient_accumulation_steps for a single GPU.")
        parser.add_argument("--learning_rate", default=1e-3, type=float,
                            help="initial learning rate.")
        parser.add_argument(
            "--num_valid", default=20, type=int,
            help="Run validation X times during training and checkpoint.")
        parser.add_argument(
            "--min_valid_steps", default=100, type=int,
            help="minimum #steps between two validation runs")
        parser.add_argument(
            "--save_steps_ratio", default=0.01, type=float,
            help="save every 0.01*global steps to resume after preemption,"
                 "not used for checkpointing.")
        parser.add_argument("--num_train_epochs", default=50, type=int,
                            help="Total #training epochs.")
        parser.add_argument("--optim", default="adamw",
                            choices=["adam", "adamax", "adamw"],
                            help="optimizer")
        parser.add_argument("--betas", default=[0.9, 0.98],
                            nargs=2, help="beta for adam optimizer")
        parser.add_argument("--decay", default="linear",
                            choices=["linear", "invsqrt"],
                            help="learning rate decay method")
        parser.add_argument("--dropout", default=0.1, type=float,
                            help="tune dropout regularization")
        parser.add_argument("--weight_decay", default=1e-3, type=float,
                            help="weight decay (L2) regularization")
        parser.add_argument("--grad_norm", default=2.0, type=float,
                            help="gradient clipping (-1 for no clipping)")
        parser.add_argument(
            "--warmup_ratio", default=0.1, type=float,
            help="to perform linear learning rate warmup for. (invsqrt decay)")
        parser.add_argument("--lr_mul", default=1.0, type=float,
                            help="lr_mul for model")
        parser.add_argument(
            "--lr_mul_prefix", default="", type=str, help="lr_mul param prefix for model")
        parser.add_argument("--lr_decay", default="linear", choices=["linear", "invsqrt", "multi_step", "constant"],
                            help="learning rate decay method")
        parser.add_argument("--step_decay_epochs", type=int,
                            nargs="+", help="multi_step decay epochs")
        # bert parameters
        parser.add_argument("--plm_optim", default="adamw", type=str,
                            choices=["adam", "adamax", "adamw", "sgd"],
                            help="optimizer for Bert")
        parser.add_argument("--plm_learning_rate", default=1e-5, type=float,
                            help="learning rate for Bert")
        parser.add_argument("--plm_weight_decay", default=1e-3, type=float,
                            help="weight decay for Bert")
        parser.add_argument("--sgd_momentum", default=0.9, type=float,
                            help="momentum for Bert")
        parser.add_argument("--plm_lr_mul", default=1.0, type=float,
                            help="bert_lr_mul for Bert")
        parser.add_argument(
            "--plm_lr_mul_prefix", default="grid_encoder", type=str,
            help="lr_mul param prefix for Bert")
        parser.add_argument("--plm_lr_decay", default="linear",
                            choices=["linear", "invsqrt", "multi_step",
                                     "constant"],
                            help="learning rate decay method")
        parser.add_argument("--bert_step_decay_epochs", type=int,
                            nargs="+", help="Bert multi_step decay epochs")
        parser.add_argument(
            "--freeze_plm", default=0, choices=[0, 1], type=int,
            help="freeze Bert by setting the requires_grad=False for Bert parameters.")

        parser.add_argument("--num_layers", default=1, type=int, help="the number of layers in lstm")
        parser.add_argument("--bidirect", default=1, type=int, choices=[0, 1], help="whether bidirect for lstm or not")
        
        # model arch # checkpoint
        parser.add_argument("--e2e_weights_path", type=str,
                            help="path to e2e model weights")
        parser.add_argument("--plm_weights_path", type=str, default="./data/bart-base",
                            help="path to BERT weights, only use for finetuning")
        
        # model parameters
        
        parser.add_argument("--hidden_size", default=128, type=int,
                            help="hidden size for model.")
        parser.add_argument("--plm_output_size", default=768, type=int,
                            help="plm_output_size for model.")
        parser.add_argument("--max_AC_num", default=12, type=int)
        parser.add_argument("--max_pair_dis", default=9, type=int)

        parser.add_argument("--graph_type", default='RGAT', type=str)
        parser.add_argument("--edge_norm", default='1', type=str2bool)
        parser.add_argument("--graph_layer", default=1, type=int, help="")
        parser.add_argument("--residual", default='1', type=str2bool, help="")
        parser.add_argument("--graph_head", default=1, type=int, help="")
        parser.add_argument("--mask_type", default="", type=str, help="")
        parser.add_argument("--init_pmt_type", default="", type=str, help="")
        parser.add_argument("--prompt_rep_type", default="", type=str, help="")
        parser.add_argument("--context_representation", default="", type=str, help="")
        parser.add_argument("--pos_type", default=1, type=int, help="")

        parser.add_argument("--loss_weight_type", default="", type=str, help="")
        parser.add_argument("--use_mlm", default='false', type=str2bool)
        parser.add_argument("--mlm_weight", default=1.0, type=float, help="")
        parser.add_argument("--actc_weight", default=1.0, type=float, help="")
        parser.add_argument("--ari_weight", default=1.0, type=float, help="")
        parser.add_argument("--artc_weight", default=1.0, type=float, help="")
        parser.add_argument("--pmtloss_type", default='CIR', type=str, help="")

        
        # inference only, please include substring `inference'
        # in the option to avoid been overwrite by loaded options,
        # see start_inference() in run_vqa_w_hvd.py
        parser.add_argument("--inference_model_step", default=-1, type=int,
                            help="pretrained model checkpoint step")
        parser.add_argument(
            "--do_inference", default=0, type=int, choices=[0, 1],
            help="perform inference run. 0: disable, 1 enable")
        parser.add_argument(
            "--inference_split", default="val",
            help="For val, the data should have ground-truth associated it."
                 "For test*, the data comes with no ground-truth.")
        parser.add_argument("--data_path", type=str,
                            help="path to data file for train")
        parser.add_argument("--split_test_file_path", type=str,
                            help="path to data file for test")

        # device parameters
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument(
            "--fp16", type=int, choices=[0, 1], default=0,
            help="Use 16-bit float precision instead of 32-bit."
                 "0: disable, 1 enable")
        parser.add_argument("--n_workers", type=int, default=4,
                            help="#workers for data loading")
        parser.add_argument("--pin_mem", type=int, choices=[0, 1], default=1,
                            help="pin memory. 0: disable, 1 enable")

        parser.add_argument("--model_version", type=str, default='1')

        # can use config files, will only overwrite unset parameters
        parser.add_argument("--config", help="JSON config files")
        self.parser = parser

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        args = parse_with_config(parsed_args)

        # convert to all [0, 1] options to bool, including these task specific ones
        zero_one_options = [
            "fp16", "pin_mem", "use_itm", "", "debug", "freeze_cnn",
            "do_inference", "bidirect",
        ]
        for option in zero_one_options:
            if hasattr(args, option):
                setattr(args, option, bool(getattr(args, option)))

        # basic checks
        # This is handled at TrainingRestorer
        # if exists(args.output_dir) and os.listdir(args.output_dir):
        #     raise ValueError(f"Output directory ({args.output_dir}) "
        #                      f"already exists and is not empty.")
        if args.bert_step_decay_epochs and args.bert_lr_decay != "multi_step":
            Warning(
                f"--bert_step_decay_epochs set to {args.bert_step_decay_epochs}"
                f"but will not be effective, as --bert_lr_decay set to be {args.bert_lr_decay}")
        if args.step_decay_epochs and args.decay != "multi_step":
            Warning(
                f"--step_decay_epochs epochs set to {args.step_decay_epochs}"
                f"but will not be effective, as --decay set to be {args.decay}")

        assert args.gradient_accumulation_steps >= 1, \
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps} "

        assert 1 >= args.data_ratio > 0, \
            f"--data_ratio should be [1.0, 0), but get {args.data_ratio}"

        return args

    def get_pe_args(self):
        args = self.parse_args()
        args.ac_type2id = {"MajorClaim": 0, "Claim": 1, "Premise": 2}
        args.para_type2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}
        args.rel_type2id = {"Support": 0, "Attack": 1}
        args.AC_type_label_num = 3
        args.link_type_label_num = 2
        return args

    def get_cdcp_args(self):
        args = self.parse_args()
        args.ac_type2id = {"value": 0, "policy": 1, "testimony": 2, "fact": 3, "reference": 4}
        args.rel_type2id = {"reason": 0, "evidence": 1}
        args.para_type2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}
        args.AC_type_label_num = 5
        args.link_type_label_num = 2
        return args


shared_configs = SharedConfigs()
