from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, BartForConditionalGeneration
from transformers.modeling_outputs import (
    MaskedLMOutput,
)
from transformers import DataCollatorForLanguageModeling
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import BartTokenizer, BartConfig, AutoTokenizer
import os
from .loss import multilabel_categorical_crossentropy
from .graph import GraphEncoder, RGAT, RGCN
from .attention import CrossAttention
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score
from models.pos_map_cdcp import ar_idx_p_matrix, ac_type_idx_p_matrix, ar_type_idx_p_matrix, pair2sequence


virual_token_num_map = {2: 8, 3: 13, 4: 20, 5: 29, 6: 40, 7: 53, 8: 68, 9: 85, 10: 104, 11: 125, 12: 148, 13: 173,
                        14: 198, 15: 223, 16: 248, 17: 273, 18: 298, 19: 323, 20: 348, 21: 373, 22: 398, 23: 423,
                        24: 448, 25: 473, 26: 498, 27: 523, 28: 548}
pair_num_map = {1: 0, 2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 55, 12: 66, 13: 78, 14: 90,
                15: 102, 16: 114, 17: 126, 18: 138, 19: 150, 20: 162, 21: 174, 22: 186, 23: 198, 24: 210, 25: 222,
                26: 234, 27: 246, 28: 258}
class RGATEmbedding(nn.Module):
    def __init__(self, config, embedding, label_embedding, prompt_embedding, prefix_embedding, tokenizer=None):
        super(RGATEmbedding, self).__init__()
        self.graph_type = config.graph_type
        self.num_labels = config.num_labels
        self.rel_num = config.rel_num
        self.graph_layer = config.graph_layer
        self.dropout = config.dropout
        self.edge_norm = config.edge_norm
        self.residual = config.residual
        self.plm_output_size = config.plm_output_size
        if self.graph_type == 'RGAT':
            self.graph = RGAT(self.rel_num, self.plm_output_size, self.plm_output_size, self.graph_layer, self.dropout,
                              edge_norm=self.edge_norm, residual=self.residual)
        elif self.graph_type == 'RGCN':
            self.graph = RGCN(self.rel_num, self.plm_output_size, self.plm_output_size, self.graph_layer, self.dropout,
                              edge_norm=self.edge_norm, residual=self.residual)
        else:
            raise ValueError
        self.padding_idx = tokenizer.pad_token_id
        self.original_embedding = embedding
        new_embedding = torch.cat(
            [torch.zeros(1, label_embedding.size(-1), device=label_embedding.device, dtype=label_embedding.dtype),
             label_embedding, prompt_embedding, prefix_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.prompt_idx = self.original_embedding.num_embeddings + label_embedding.size(0)

    @property
    def weight(self):
        def foo():
            # # label prompt MASK
            # edge_features = self.new_embedding.weight[1+self.num_labels:, :]
            # if self.graph_type != '':
            #     # label prompt
            #     edge_features = edge_features[:-1, :]
            #     edge_features = self.graph(edge_features)
            #     edge_features = torch.cat(
            #         [self.new_embedding.weight[1:self.num_labels, :], edge_features, self.new_embedding.weight[-1:, :]], dim=0)
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)
        return foo

    @property
    def raw_weight(self):
        def foo():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)

        return foo

    def forward(self, x, adjs=None, span_num=None, context_rep=None):
        # print("self.weight()[-1]", self.weight()[-1])  # see whether the parameters have been updated or not

        if adjs == None and span_num == None:
            y = F.embedding(x, self.weight(), self.padding_idx)
        else:
            if context_rep == None:
                if span_num == 1:
                    y = F.embedding(x, self.weight(), self.padding_idx)
                else:
                    prompt_ids = torch.ge(x, self.prompt_idx)
                    # print("span_num", span_num)
                    # print("prompt_ids", prompt_ids.size(), self.prompt_idx, prompt_ids.type())
                    e = F.embedding(x, self.weight(), self.padding_idx)
                    y = torch.zeros_like(e, device=e.device)
                    # print("e", e.size())
                    prompt_ids = prompt_ids.unsqueeze(-1).expand(-1, -1, e.size(-1))
                    prompt_embed = torch.masked_select(e, prompt_ids).view(e.size(0), -1, e.size(-1))
                    # print("prompt_embed", prompt_embed.size(), span_num, virual_token_num_map[span_num])
                    assert prompt_embed.size(1) == virual_token_num_map[span_num], print("prompt_embed", prompt_embed.size(), span_num, virual_token_num_map[span_num])
                    assert adjs.size(2) == virual_token_num_map[span_num], print("adjs.size()", adjs.size())
                    prompt_embed = self.graph(prompt_embed, adjs)

                    # print("prompt_ids", prompt_ids.size())
                    # print("prompt_embed", prompt_embed.size())
                    y[~prompt_ids] = e[~prompt_ids]
                    y[prompt_ids] = prompt_embed.view(-1)

                    # e = torch.masked_fill(e, prompt_ids, prompt_embed)
            else:
                if span_num == 1:
                    prompt_ids = torch.ge(x, self.prompt_idx)
                    e = F.embedding(x, self.weight(), self.padding_idx)
                    y = torch.zeros_like(e, device=e.device)
                    prompt_ids = prompt_ids.unsqueeze(-1).expand(-1, -1, e.size(-1))
                    prompt_embed = torch.masked_select(e, prompt_ids).view(e.size(0), -1, e.size(
                        -1)) + context_rep  # [batch_size, span_num+pair_num*2+1, dim]
                    y[~prompt_ids] = e[~prompt_ids]
                    y[prompt_ids] = prompt_embed.view(-1)
                else:
                    prompt_ids = torch.ge(x, self.prompt_idx)
                    e = F.embedding(x, self.weight(), self.padding_idx)
                    y = torch.zeros_like(e, device=e.device)
                    prompt_ids = prompt_ids.unsqueeze(-1).expand(-1, -1, e.size(-1))
                    prompt_embed = torch.masked_select(e, prompt_ids).view(e.size(0), -1, e.size(-1)) + context_rep # [batch_size, span_num+pair_num*2+1, dim]
                    prompt_embed = self.graph(prompt_embed, adjs)
                    y[~prompt_ids] = e[~prompt_ids]
                    y[prompt_ids] = prompt_embed.view(-1)

        return y


class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class BartForPrompt(BartPretrainedModel):
    # base_model_prefix = "model"
    # _keys_to_ignore_on_load_missing = [
    #     r"final_logits_bias",
    #     r"lm_head.weight",
    #     "encoder.embed_tokens.weight",
    #     "decoder.embed_tokens.weight",
    # ]

    def __init__(self, config, context_representation='decoder', prompt_rep_type="graph", use_mlm=False):
        super().__init__(config)

        self.model = BartModel(config)
        self.context_representation = context_representation
        self.use_mlm = use_mlm
        self.prompt_rep_type = prompt_rep_type

        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings)
        self.bias = nn.Parameter(torch.zeros(self.model.shared.num_embeddings))
        self.lm_head.bias = self.bias
        # self.final_logits_bias = nn.Parameter(torch.zeros((1, self.model.shared.num_embeddings)))
        # self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # self.lm_head.bias = self.final_logits_bias

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def resize_token_embeddings(self, new_num_tokens):
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        # self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # def _resize_final_logits_bias(self, new_num_tokens):
    #     old_num_tokens = self.final_logits_bias.shape[-1]
    #     if new_num_tokens <= old_num_tokens:
    #         new_bias = self.final_logits_bias[:, :new_num_tokens]
    #     else:
    #         extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
    #         new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
    #     self.register_buffer("final_logits_bias", new_bias)

    def forward(
        self,
        input_ids = None,
        AC_spans_list = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        corss_attention_mask = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        adjs=None,
        span_num=None
    ):

        if self.context_representation == 'decoder':
            context_outputs = self.model(
                input_ids = input_ids,
                attention_mask= attention_mask,
                return_dict=True,
            )
            decoder_context = context_outputs.encoder_last_hidden_state
            # context_outputs = context_outputs.last_hidden_state
        else:
            context_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            context_outputs = context_outputs.last_hidden_state
            decoder_context = context_outputs

        if self.prompt_rep_type == "none":
            decoder_inputs_embeds = self.model.shared(decoder_input_ids)
        elif self.prompt_rep_type == "none2":
            prompt_context_rep = []
            temp_prompt_context_rep = []
            for batch_i, AC_spans in enumerate(AC_spans_list):
                batch_i_rep = [decoder_context[batch_i, 1:-1].mean(0)]
                batch_i_rep.append(torch.zeros_like(batch_i_rep[-1], device=batch_i_rep[-1].device))

                tmp_batch_i_rep = [decoder_context[batch_i, 1:-1].mean(0)]
                for span in AC_spans:
                    tmp_batch_i_rep.append(decoder_context[batch_i, span[0]:span[1] + 1].mean(0))
                    batch_i_rep.append(tmp_batch_i_rep[-1])
                batch_i_rep = torch.stack(batch_i_rep, 0)  # [span_num+1, dim]
                tmp_batch_i_rep = torch.stack(tmp_batch_i_rep, 0)
                prompt_context_rep.append(batch_i_rep)
                temp_prompt_context_rep.append(tmp_batch_i_rep)
            prompt_context_rep = torch.stack(prompt_context_rep, 0)  # [batch_size, 2*span_num+1, dim]
            temp_prompt_context_rep = torch.stack(temp_prompt_context_rep, 0)  # [batch_size, span_num+1, dim]
            if span_num > 1:
                temp_context_rep = [torch.zeros((prompt_context_rep.size(0), 1, prompt_context_rep.size(-1)),
                                                device=prompt_context_rep.device)]
                for pair in list(pair2sequence[span_num]):
                    temp_context_rep.append(torch.stack([temp_prompt_context_rep[:, pair[0] + 1],
                                                         temp_prompt_context_rep[:, pair[1] + 1]], dim=1).mean(1, keepdim=True))

                prompt_context_rep = torch.cat([prompt_context_rep] + temp_context_rep + temp_context_rep, 1)  # [batch_size, span_num+1+pair_num*2, dim]
            prompt_context_rep = torch.cat([prompt_context_rep,
                                            torch.zeros((prompt_context_rep.size(0), 1,
                                                         prompt_context_rep.size(-1)),
                                                        device=prompt_context_rep.device)], dim=1)
            decoder_inputs_embeds = self.model.shared(decoder_input_ids) + prompt_context_rep

        elif self.prompt_rep_type == "graph":
            decoder_inputs_embeds = self.model.shared(decoder_input_ids, adjs, span_num)
        elif self.prompt_rep_type == "graph2":
            prompt_context_rep = []
            temp_prompt_context_rep = []
            for batch_i, AC_spans in enumerate(AC_spans_list):
                batch_i_rep = [decoder_context[batch_i, 1:-1].mean(0)]
                batch_i_rep.append(torch.zeros_like(batch_i_rep[-1], device=batch_i_rep[-1].device))

                tmp_batch_i_rep = [decoder_context[batch_i, 1:-1].mean(0)]
                for span in AC_spans:
                    tmp_batch_i_rep.append(decoder_context[batch_i, span[0]:span[1] + 1].mean(0))
                    batch_i_rep.append(tmp_batch_i_rep[-1])
                batch_i_rep = torch.stack(batch_i_rep, 0)  # [span_num+1, dim]
                tmp_batch_i_rep = torch.stack(tmp_batch_i_rep, 0)
                prompt_context_rep.append(batch_i_rep)
                temp_prompt_context_rep.append(tmp_batch_i_rep)
            prompt_context_rep = torch.stack(prompt_context_rep, 0)  # [batch_size, 2*span_num+1, dim]
            temp_prompt_context_rep = torch.stack(temp_prompt_context_rep, 0)  # [batch_size, span_num+1, dim]
            if span_num > 1:
                temp_context_rep = [torch.zeros((prompt_context_rep.size(0), 1, prompt_context_rep.size(-1)),
                                                device=prompt_context_rep.device)]
                for pair in list(pair2sequence[span_num]):
                    temp_context_rep.append(torch.stack([temp_prompt_context_rep[:, pair[0] + 1],
                                                         temp_prompt_context_rep[:, pair[1] + 1]], dim=1).mean(1,
                                                                                                               keepdim=True))

                prompt_context_rep = torch.cat([prompt_context_rep] + temp_context_rep + temp_context_rep,
                                               1)  # [batch_size, span_num+1+pair_num*2, dim]
            prompt_context_rep = torch.cat([prompt_context_rep,
                                            torch.zeros((prompt_context_rep.size(0), 1,
                                                         prompt_context_rep.size(-1)),
                                                        device=prompt_context_rep.device)], dim=1)

            decoder_inputs_embeds = self.model.shared(decoder_input_ids, adjs, span_num, prompt_context_rep)

        elif self.prompt_rep_type == "graph3":
            prompt_context_rep = []
            temp_prompt_context_rep = []
            for batch_i, AC_spans in enumerate(AC_spans_list):
                batch_i_rep = [decoder_context[batch_i, 1:-1].mean(0)]
                batch_i_rep.append(torch.zeros_like(batch_i_rep[-1], device=batch_i_rep[-1].device))

                tmp_batch_i_rep = [decoder_context[batch_i, 1:-1].mean(0)]
                for span in AC_spans:
                    tmp_batch_i_rep.append(decoder_context[batch_i, span[0]:span[1] + 1].mean(0))
                    batch_i_rep.append(tmp_batch_i_rep[-1])
                batch_i_rep = torch.stack(batch_i_rep, 0)  # [span_num+1, dim]
                tmp_batch_i_rep = torch.stack(tmp_batch_i_rep, 0)
                prompt_context_rep.append(batch_i_rep)
                temp_prompt_context_rep.append(tmp_batch_i_rep)
            prompt_context_rep = torch.stack(prompt_context_rep, 0)  # [batch_size, 2*span_num+1, dim]
            temp_prompt_context_rep = torch.stack(temp_prompt_context_rep, 0)  # [batch_size, span_num+1, dim]
            if span_num > 1:
                temp_context_rep = [torch.zeros((prompt_context_rep.size(0), 1, prompt_context_rep.size(-1)),
                                                device=prompt_context_rep.device)]
                for pair in list(pair2sequence[span_num]):
                    temp_context_rep.append(torch.stack([temp_prompt_context_rep[:, pair[0] + 1],
                                                         temp_prompt_context_rep[:, pair[1] + 1]], dim=1).mean(1, keepdim=True))
                prompt_context_rep = torch.cat([prompt_context_rep] + temp_context_rep + temp_context_rep, 1)  # [batch_size, span_num+1+pair_num*2, dim]
            prompt_context_rep = torch.cat([prompt_context_rep,
                                            torch.zeros((prompt_context_rep.size(0), 1,
                                                         prompt_context_rep.size(-1)),
                                                        device=prompt_context_rep.device)], dim=1)
            decoder_inputs_embeds = self.model.shared(decoder_input_ids, adjs, span_num) + prompt_context_rep

        else:
            raise ValueError


        decoder_prompt_outputs = self.model.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=decoder_context,
            encoder_attention_mask=corss_attention_mask,
        )
        context_scores = None
        if self.use_mlm:
            assert self.use_mlm == False, print("self.use_mlm", self.use_mlm, type(self.use_mlm))
            context_scores = self.lm_head(decoder_context)

        decoder_prompt_outputs = decoder_prompt_outputs.last_hidden_state  # [bs, prompt_len, H]

        prompt_scores = self.lm_head(decoder_prompt_outputs) # [bs, prompt_len, vocab_size]

        return context_scores, prompt_scores


class JointPrompt(nn.Module):

    def __init__(self, config, **kwargs):
        # super().__init__(config)
        super(JointPrompt, self).__init__()

        self.config = config
        self.special_tokens = ['<essay>', '<para-conclusion>', '<para-body>', '<para-intro>', '<ac>',
                               '</essay>', '</para-conclusion>', '</para-body>', '</para-intro>', '</ac>']
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_weights_path, add_special_tokens=True)
        self.tokenizer.add_tokens(self.special_tokens)  ############为 bart 设置特殊字符

        self.bartprompt = BartForPrompt.from_pretrained(config.plm_weights_path,
                                                        context_representation=config.context_representation,
                                                        prompt_rep_type=config.prompt_rep_type,
                                                        use_mlm=config.use_mlm)
        self.bartprompt.resize_token_embeddings(len(self.tokenizer)) # ???

        self.device = self.bartprompt.device

        self.num_labels = config.num_labels
        self.actc_class_bias = nn.Parameter(torch.zeros(5, dtype=torch.float32))
        self.ari_class_bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.artc_class_bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.actc_class_bias, -bound, bound)
        nn.init.uniform_(self.ari_class_bias, -bound, bound)
        nn.init.uniform_(self.artc_class_bias, -bound, bound)
        self.vocab_size = len(self.tokenizer)
        self.max_AC_num = config.max_AC_num
        self.max_pair_num = 258
        self.pad_token_id = self.tokenizer.pad_token_id
        self.init_pmt_type = config.init_pmt_type
        self.use_mlm = config.use_mlm
        self.loss_weight_type = config.loss_weight_type
        if self.loss_weight_type == '':
            pass
        elif self.loss_weight_type == 'fixed':
            self.actc_weight = config.actc_weight
            self.ari_weight = config.ari_weight
            self.artc_weight = config.artc_weight
            if self.use_mlm:
                self.mlm_weight = config.mlm_weight
        elif self.loss_weight_type == 'learned':
            self.actc_weight = nn.Parameter(torch.ones(1, requires_grad=True))
            self.ari_weight = nn.Parameter(torch.ones(1, requires_grad=True))
            self.artc_weight = nn.Parameter(torch.ones(1, requires_grad=True))
            if self.use_mlm:
                self.mlm_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        else:
            raise ValueError

        # self.apply(self.init_weights)

        self.init_embedding()

    def get_output_embeddings(self):
        return self.bartprompt.lm_head

    # def get_output_embedding_bias(self):
    #     return self.bartprompt.final_logits_bias

    def set_output_embeddings(self, new_embeddings):
        self.bartprompt.lm_head = new_embeddings

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_embedding(self):
        label_dict = {0: 'value', 1: 'policy', 2: 'testimony', 3: 'fact', 4: 'reference', 5: 'no relation',
                      6: 'relation', 7: "reason", 8: "evidence"}
        label_dict = {i: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(v)) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.bartprompt.get_input_embeddings()
        # print("input_embeds", type(input_embeds))
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        label_emb = torch.stack(label_emb)
        prefix = torch.stack([label_emb[:5].data.mean(0), label_emb[5:7].data.mean(0), label_emb[7:].data.mean(0)])

        # print("prefix", prefix)

        # prompt TODO initilize the embeddings with prior label information
        if self.init_pmt_type == "none":
            prompt_embedding = nn.Embedding(self.max_AC_num + self.max_pair_num * 2 + 1 + 1,
                                            input_embeds.weight.size(1), 0)

            self.init_weights(prompt_embedding)
        else:
            prompt_tensor = torch.ones((self.max_AC_num + self.max_pair_num * 2 + 1 + 1, input_embeds.weight.size(1)),
                                       device=self.device)

            prompt_tensor[1:2] = input_embeds.weight.data[0:1]

            if self.init_pmt_type == "type":
                prompt_tensor[2:self.max_AC_num + 2] = label_emb[:5].data.mean(0).repeat(self.max_AC_num, 1)
                prompt_tensor[self.max_AC_num + 2: self.max_AC_num + 2 + self.max_pair_num] = label_emb[
                                                                                              5:7].data.mean(
                    0).repeat(self.max_pair_num, 1)
                prompt_tensor[
                self.max_AC_num + 2 + self.max_pair_num: self.max_AC_num + 2 + self.max_pair_num * 2] = label_emb[
                                                                                                        7:].data.mean(
                    0).repeat(self.max_pair_num, 1)
            elif self.init_pmt_type == "fre":
                prompt_tensor[2:self.max_AC_num + 2] = torch.matmul(
                    torch.tensor([[2158 / (2158 + 810 + 1026 + 746 + 32), 810 / (2158 + 810 + 1026 + 746 + 32),
                                   1026 / (2158 + 810 + 1026 + 746 + 32), 746 / (2158 + 810 + 1026 + 746 + 32),
                                   32 / (2158 + 810 + 1026 + 746 + 32)]],
                                 device=self.device), label_emb[:5].data).repeat(self.max_AC_num, 1)
                prompt_tensor[self.max_AC_num + 2: self.max_AC_num + 2 + self.max_pair_num] = torch.matmul(
                    torch.tensor([[18455 / (18455 + 1352), 1352 / (18455 + 1352)]],
                                 device=self.device), label_emb[5:7].data).repeat(self.max_pair_num, 1)
                prompt_tensor[
                self.max_AC_num + 2 + self.max_pair_num: self.max_AC_num + 2 + self.max_pair_num * 2] = torch.matmul(
                    torch.tensor([[1306 / (1306 + 46), 46 / (1306 + 46)]],
                                 device=self.device), label_emb[7:].data).repeat(self.max_pair_num, 1)
            elif self.init_pmt_type == "pos_fre":
                prompt_tensor[2:self.max_AC_num + 2] = torch.matmul(
                    torch.tensor(ac_type_idx_p_matrix, device=self.device), label_emb[:5].data)
                prompt_tensor[self.max_AC_num + 2: self.max_AC_num + 2 + self.max_pair_num] = torch.matmul(
                    torch.tensor(ar_idx_p_matrix, device=self.device), label_emb[5:7].data)
                prompt_tensor[
                self.max_AC_num + 2 + self.max_pair_num: self.max_AC_num + 2 + self.max_pair_num * 2] = torch.matmul(
                    torch.tensor(ar_type_idx_p_matrix, device=self.device), label_emb[7:].data)
            else:
                raise ValueError

            prompt_embedding = nn.Embedding.from_pretrained(prompt_tensor, freeze=False, padding_idx=0)

        # label prompt mask
        prompt_emb = prompt_embedding.weight[1:, :]
        self.embedding = RGATEmbedding(self.config, input_embeds, label_emb, prompt_emb, prefix, self.tokenizer)
        self.bartprompt.set_input_embeddings(self.embedding)

        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        output_embeddings.weight = self.embedding.raw_weight
        vocab_size = output_embeddings.bias.size(0)
        assert self.vocab_size == vocab_size, print("self.vocab_size", self.vocab_size, "vocab_size", vocab_size)
        output_embeddings.bias.data = F.pad(output_embeddings.bias.data,
                                            (0, self.embedding.size - output_embeddings.bias.shape[0],), "constant", 0)
        self.set_output_embeddings(output_embeddings)

    def forward(
            self,
            input_ids=None,
            AC_spans_list=None,
            prompt_ids=None,
            adjs=None,
            span_num=None,
            attention_mask=None,
            decoder_attention_mask=None,
            corss_attention_mask=None,
            labels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        # torch.ge(x, self.prompt_idx) & torch.lt(x, self.size - 3)
        predict_pos =  (prompt_ids > (self.embedding.prompt_idx)) & (prompt_ids < (self.embedding.size - 3)) # [bs, span_num+pair_num*2] and 3 represents the num of task token
        single_labels = input_ids.masked_fill(input_ids == self.pad_token_id, -100)
        if self.training:
            if self.use_mlm:
                enable_mask = input_ids < self.tokenizer.vocab_size - 1
                random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
                input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
                random_ids = torch.randint_like(input_ids, 4, self.vocab_size-1)
                mlm_mask = random_mask > 0.985
                input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
                mlm_mask = random_mask < 0.85
                single_labels = single_labels.masked_fill(mlm_mask, -100)

        # setting the input_ids = None and using the inputs_embeds
        # prompt_embeds = self.embedding(prompt_ids, adjs, span_num)
        # print("prompt_embeds", prompt_embeds.size())

        context_scores, prompt_scores = self.bartprompt(
            input_ids=input_ids,
            AC_spans_list=AC_spans_list,
            attention_mask=attention_mask,
            decoder_input_ids=prompt_ids,
            decoder_attention_mask=decoder_attention_mask,
            # f=prompt_embeds,
            corss_attention_mask=corss_attention_mask,
            adjs=adjs,
            span_num=span_num
        )

        # sequence_output = outputs[0]
        # prediction_scores = self.lm_head(sequence_output)
        # print("outputs", type(outputs))
        # print("prediction_scores", prediction_scores.size())

        masked_lm_loss = None
        actc_loss = None
        ari_loss = None
        artc_loss = None
        batch_size = input_ids.size(0)

        if labels is not None:
            if self.use_mlm:
                masked_lm_loss = F.cross_entropy(context_scores.view(-1, context_scores.size(-1)),
                                          single_labels.view(-1), reduction="none") # -100 index = padding token

            logits = prompt_scores.masked_select(
                predict_pos.unsqueeze(-1).expand(-1, -1, prompt_scores.size(-1))).view(batch_size, -1, prompt_scores.size(-1))
            # print("predict_pos", predict_pos.sum(-1))
            actc_logits = logits[:, :span_num,
                          self.vocab_size:self.vocab_size + 5] + self.actc_class_bias  # [bs, span_num, 5]
            actc_loss = F.cross_entropy(actc_logits.view(-1, 5), labels[0], reduction='none')

            if span_num > 1:
                ari_logits = logits[:, span_num:span_num + pair_num_map[span_num],
                             self.vocab_size + 5:self.vocab_size + 7] + self.ari_class_bias  # [bs, pair_num, 2]
                artc_logits = logits[:, span_num + pair_num_map[span_num]:,
                              self.vocab_size + 7:self.vocab_size + 9] + self.artc_class_bias  # [bs, pair_num, 2]
                # print("ari_logits", ari_logits.size())
                # print("labels[1]", labels[1].size())
                # print("artc_logits", artc_logits.size())
                # print("labels[2]", labels[2].size())

                ari_loss = F.cross_entropy(ari_logits.view(-1, 2), labels[1], reduction='none')
                artc_loss = F.cross_entropy(artc_logits.view(-1, 2), labels[2], reduction='none')

            if self.loss_weight_type == 'fixed':
                actc_loss = self.actc_weight * actc_loss
                if span_num > 1:
                    ari_loss = self.ari_weight * ari_loss
                    artc_loss = self.artc_weight * artc_loss
                if self.use_mlm:
                    masked_lm_loss = self.mlm_weight * masked_lm_loss
            elif self.loss_weight_type == 'learned':
                actc_loss = 0.5 / (self.actc_weight ** 2) * actc_loss + torch.log(1 + self.actc_weight ** 2)
                if span_num > 1:
                    ari_loss = 0.5 / (self.ari_weight ** 2) * ari_loss + torch.log(1 + self.ari_weight ** 2)
                    artc_loss = 0.5 / (self.artc_weight ** 2) * artc_loss + torch.log(1 + self.artc_weight ** 2)
                if self.use_mlm:
                    masked_lm_loss = 0.5 / (self.mlm_weight ** 2) * masked_lm_loss + torch.log(1 + self.mlm_weight ** 2)

        ret = dict(
            ml_loss=masked_lm_loss,
            actc_loss=actc_loss,
            ari_loss=ari_loss,
            artc_loss=artc_loss,
            logits=prompt_scores
        )
        return ret

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def predict(self, input_ids, AC_spans_list, prompt_ids, adjs, span_num,
                attention_mask = None,
                decoder_attention_mask = None,
                corss_attention_mask = None):

        outputs = self(input_ids, AC_spans_list, prompt_ids, adjs, span_num,
                       attention_mask,
                       decoder_attention_mask,
                       corss_attention_mask)

        predict_pos = (prompt_ids > (self.embedding.prompt_idx)) & (prompt_ids < (self.embedding.size - 3))
        prediction_scores = outputs['logits']
        # print("predict_pos", predict_pos.sum(-1))
        # print("prediction_scores", prediction_scores.size())

        prediction_scores = prediction_scores.masked_select(
            predict_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(prediction_scores.size(0), -1,
                                                                                       prediction_scores.size(-1))

        # print("prediction_scores", prediction_scores.size(), span_num)
        actc_scores = prediction_scores[:, :span_num, self.vocab_size:self.vocab_size + 5] + self.actc_class_bias  # [bs, span_num, 5]
        actc_scores = actc_scores.view(-1, 5)
        if span_num > 1:
            ari_scores = prediction_scores[:, span_num:span_num + pair_num_map[span_num],
                         self.vocab_size + 5:self.vocab_size + 7] + self.ari_class_bias  # [bs, pair_num, 2]
            artc_scores = prediction_scores[:, span_num + pair_num_map[span_num]:,
                          self.vocab_size + 7:self.vocab_size + 9] + self.artc_class_bias  # [bs, pair_num, 2]

            ari_scores = ari_scores.view(-1, 2)
            artc_scores = artc_scores.view(-1, 2)

            # print("ari_scores", ari_scores.size())
            # print("artc_scores", artc_scores.size())
        else:
            ari_scores = None
            artc_scores = None

        return actc_scores, ari_scores, artc_scores
