#!/usr/bin/env python
# coding: utf-8

from transformers import RobertaModel
from torch import nn


class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.pretrained_model)
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.transform = nn.Sequential(
            nn.Linear(768, 1000),#为了和pic等长
            nn.ReLU(),
        )

    def forward(self, encoded_input):
        encoder_output = self.encoder(input_ids=encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"] )
        hidden_state = encoder_output['last_hidden_state']
        pooler_output = encoder_output['pooler_output']
        output = self.transform(pooler_output)
        return hidden_state, output

class TextClassifier(nn.Module):
    def __init__(self, args):
        super(TextClassifier, self).__init__()
        self.TextModel = TextModel(args)
        self.classifier_text = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 3),
        )
    def forward(self, batch_text=None, batch_img=None):
        _, text_output = self.TextModel(batch_text)
        text_output = self.classifier_text(text_output)
        return text_output