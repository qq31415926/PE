from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.distributions import Categorical
import torch.nn.functional as F
# from captum.attr import Saliency
class bertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bertconfig = BertConfig.from_pretrained(config.bert_path)
        self.bertconfig.num_labels = config.num_classes

        self.bertmodel = BertModel.from_pretrained(config.bert_path, config=self.bertconfig)
        self.num_labels = config.num_classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        # for name,param in self.bertmodel.named_parameters():
        #     if "embedding" in name:
        #         param.requires_grad = False

    def forward_for_IG(self, input_ids, token_type_ids):
        x = self.bertmodel.embeddings(input_ids, token_type_ids)
        return x

    def forward(self, input_ids, attention_mask=None,return_pool=False, token_type_ids=None,  labels=None,
                position_ids=None, head_mask=None,output_hidden_states=False):
        # print(input_ids.shape)
        # if len(input_ids.shape) == 1:
        #     input_ids = input_ids.unsqueeze(0)
        #     attention_mask = attention_mask.unsqueeze(0)
        outputs = self.bertmodel(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask,output_hidden_states=output_hidden_states)
        # assert len(outputs) == 3
        pooled_emb = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits_ = self.classifier(pooled_output)
        logits = F.softmax(logits_,dim=-1)
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                # print("loss:{} labels:{}".format(logits,labels))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # print("loss:")
            outputs = (loss, logits)
        if not return_pool:
            if labels is not None:
                return outputs  # (loss), logits, (hidden_states), (attentions)
            elif not output_hidden_states:
                return logits
            else:
                return logits,outputs[2]
        else:
            return pooled_emb, pooled_output