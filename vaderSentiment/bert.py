from transformers import AutoTokenizer, AutoModel
import os
import torch
import numpy as np
from torch import nn

model1 = "finiteautomata/bertweet-base-emotion-analysis"
model2 = "nghuyong/ernie-2.0-en"



tokenizer1 = AutoTokenizer.from_pretrained(model1)

tokenizer2 = AutoTokenizer.from_pretrained(model2)

# Valence_M

def bert_tokenizer(word):
    texts1 = tokenizer1(str(word).lower(),
                              padding='max_length', max_length=8, truncation=True,
                              return_tensors="pt")
    texts2 = tokenizer2(str(word).lower(),
                              padding='max_length', max_length=8, truncation=True,
                              return_tensors="pt")
    return texts1, texts2




class BertRegression(nn.Module):

    def __init__(self, dropout=0.1, hidden_dim_valence=1536, hidden_dim_arousal = 1536, hidden_dim_dominance = 1536):

        super(BertRegression, self).__init__()

        self.bert1 = AutoModel.from_pretrained(model1)
        self.bert2 = AutoModel.from_pretrained(model2)

        self.valence = nn.Linear(hidden_dim_valence, 1)
        self.arousal = nn.Linear(hidden_dim_arousal, 1)
        self.dominance = nn.Linear(hidden_dim_dominance, 1)
        self.aoa = nn.Linear(hidden_dim_dominance, 1)
        self.concreteness = nn.Linear(hidden_dim_dominance, 1)
        self.l_1_valence = nn.Linear(hidden_dim_valence, hidden_dim_valence)
        self.l_1_arousal = nn.Linear(hidden_dim_arousal, hidden_dim_arousal)
        self.l_1_dominance = nn.Linear(hidden_dim_dominance, hidden_dim_dominance)
        self.l_1_aoa = nn.Linear(hidden_dim_dominance, hidden_dim_dominance)
        self.l_1_concreteness = nn.Linear(hidden_dim_dominance, hidden_dim_dominance)

        self.layer_norm = nn.LayerNorm(hidden_dim_valence)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id1, mask1, input_id3, mask3):
        _, y = self.bert1(input_ids = input_id1, attention_mask=mask1, return_dict=False)
        _, z = self.bert2(input_ids = input_id3, attention_mask=mask3, return_dict=False)
        x = torch.cat((y, z), dim=1)
        x = self.dropout(x)


        valence_all = self.dropout(self.relu(self.layer_norm(self.l_1_valence(x) + x)))
        valence = self.sigmoid(self.valence(valence_all))

        arousal_all = self.dropout(self.relu(self.layer_norm(self.l_1_arousal(x) + x)))
        arousal = self.sigmoid(self.arousal(arousal_all))

        dominance_all = self.dropout(self.relu(self.layer_norm(self.l_1_dominance(x) + x)))
        dominance = self.sigmoid(self.dominance(dominance_all))

        aoa_all = self.dropout(self.relu(self.layer_norm(self.l_1_aoa(x) + x)))
        aoa = self.sigmoid(self.aoa(aoa_all))

        concreteness_all = self.dropout(self.relu(self.layer_norm(self.l_1_concreteness(x) + x)))
        concreteness = self.sigmoid(self.concreteness(concreteness_all))

        return valence, dominance, arousal, aoa, concreteness

#
#
# class BertRegression(nn.Module):
#
#     def __init__(self, dropout=0.1, hidden_dim_valence=1536):
#
#         super(BertRegression, self).__init__()
#
#         self.bert1 = AutoModel.from_pretrained(model1)
#         self.bert2 = AutoModel.from_pretrained(model2)
#
#         self.valence = nn.Linear(hidden_dim_valence, 1)
#
#         self.l_1_valence = nn.Linear(hidden_dim_valence, hidden_dim_valence)
#
#
#         self.layer_norm = nn.LayerNorm(hidden_dim_valence)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input_id1, mask1, input_id3, mask3):
#         _, y = self.bert1(input_ids = input_id1, attention_mask=mask1, return_dict=False)
#         _, z = self.bert2(input_ids = input_id3, attention_mask=mask3, return_dict=False)
#         x = torch.cat((y, z), dim=1)
#         x = self.dropout(x)
#
#
#         valence_all = self.dropout(self.relu(self.layer_norm(self.l_1_valence(x) + x)))
#         valence = self.sigmoid(self.valence(valence_all))
#
#
#
#         return valence
