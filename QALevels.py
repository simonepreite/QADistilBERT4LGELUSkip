import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from Transformers import (BertPreTrainedModel, BertModel, DistilBertPreTrainedModel, DistilBertModel)

class QABERT4LGELUSkip(BertPreTrainedModel):

    def __init__(self, config):
        super(QABERT4LGELUSkip, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.middleOut1 = nn.Linear(config.hidden_size, 1024)
        self.middleOut2 = nn.Linear(1024, 768)
        self.middleOut3 = nn.Linear(768, 384)
        self.qa_outputs = nn.Linear(384, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = outputs[0]
        midOut1 = self.dropout(gelu_new(self.middleOut1(sequence_output)))
        midOut2 = self.dropout(gelu_new(self.middleOut2(midOut1)))
        midOut3 = self.dropout(gelu_new(self.middleOut3(midOut2 + sequence_output)))
        
        
        logits = self.qa_outputs(midOut3)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
		
class QADistilBert4LGELUSkip(DistilBertPreTrainedModel):

    def __init__(self, config):
        super(QADistilBert4LGELUSkip, self).__init__(config)

        self.distilbert = DistilBertModel(config)
        self.middleOut1 = nn.Linear(config.dim, 1024)
        self.middleOut2 = nn.Linear(1024, 768)
        self.middleOut3 = nn.Linear(768, 384)
        self.qa_outputs = nn.Linear(384, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        midOut1 = self.dropout(gelu(self.middleOut1(hidden_states)))
        midOut2 = self.dropout(gelu(self.middleOut2(midOut1)))
        midOut3 = self.dropout(gelu(self.middleOut3(midOut2 + hidden_states)))
        
        
        logits = self.qa_outputs(midOut3)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

        outputs = (start_logits, end_logits,) + distilbert_output[1:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)