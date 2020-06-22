import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AdamW, BertConfig, BertPreTrainedModel

class BertPretrainClassification(BertPreTrainedModel):
    """
        Bert Pretrain Fine Tuning On Text Classification, add attention on all layer cls
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #print(dir(self.config))
        self.bert = BertModel(config)
        self.fc_dropout = nn.Dropout(self.config.fc_dropout)
        self.fc = nn.Linear(config.hidden_size, self.config.num_labels)
        self.attention_fc = nn.Linear(config.hidden_size, 1)

        self.init_weights()
        pass

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions
        )
        #print(outputs)

        hidden_states = outputs[2]
        all_cls = torch.cat([cls[:, 0:1, :] for i, cls in enumerate(hidden_states) if i != 0], dim=1)
        #print("all_cls shape:", all_cls.shape)
        alignment = self.attention_fc(all_cls)
        #print("alignment shape:", alignment.shape)
        attention_score = F.softmax(alignment, dim=1)
        query = torch.bmm(attention_score.transpose(1, 2), all_cls)
        query = torch.squeeze(query)
        #print("query shape:", query.shape)

        query = self.fc_dropout(query)
        logits = self.fc(query)

        logits = logits.view(-1, self.config.num_labels)

        return logits
