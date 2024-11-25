import torch
import torch.nn as nn
from transformers.models.bert import BertModel


class BertForPairClass(nn.Module):
    def __init__(self, config, bert_out_mode=None):
        super().__init__()
        self.num_classes = config.num_labels
        self.bert_out_mode = bert_out_mode
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.classifier_head = nn.Linear(config.hidden_size, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        pass

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                labels,
                ):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        bert_outputs = bert_outputs.last_hidden_state
        if self.bert_out_mode is not None:
            if self.bert_out_mode == 'cls':
                bert_outputs = bert_outputs[:, 0, :]
            elif self.bert_out_mode == 'mean':
                bert_outputs = torch.mean(bert_outputs, dim=1)
        pooled_outputs = self.dropout(bert_outputs)
        logits = self.classifier_head(pooled_outputs)
        loss = self.loss_fn(logits, labels)
        return loss, logits

