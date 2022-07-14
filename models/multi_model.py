import torch
from torch import nn
from models.text_model import TextModel,TextClassifier
from models.pic_model import PicModel,PicClassifier


class mBertModel(nn.Module):
    def __init__(self, args):
        super(mBertModel, self).__init__()
        self.TextModel = TextModel(args)
        self.PicModel = PicModel(args)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.classifier_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
        )

    def forward(self, batch_text=None, batch_img=None):
        _, text_output = self.TextModel(batch_text) 
        img_output = self.PicModel(batch_img)  
        multi_output = self.fusion_transformer(text_output, img_output)
        multi_output = self.classifier_multi(multi_output)
        return  multi_output

    def fusion_transformer(self, text_out, img_out):
        multimodal_sequence = torch.stack((text_out, img_out), dim=1)  
        return self.transformer_encoder(multimodal_sequence)


class TomBertModel(nn.Module):
    def __init__(self, args):
        super(TomBertModel, self).__init__()
        self.TextModel = TextModel(args)
        self.PicModel = PicModel(args)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.attention = nn.MultiheadAttention(embed_dim=1000, num_heads=1, batch_first=True)

        self.classifier_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
        )
        self.fc = nn.Linear(1000, 1000)

    def forward(self, batch_text=None, batch_img=None):
        _, text_output = self.TextModel(batch_text) 
        img_output = self.PicModel(batch_img)  
        multi_output = self.fusion_attention( img_output,img_output,text_output)
        multi_output = self.fusion_transformer(text_output, multi_output)
        multi_output = self.classifier_multi(multi_output)
        return  multi_output

    def fusion_transformer(self, text_out, img_out):
        multimodal_sequence = torch.stack((text_out, img_out), dim=1)  
        return self.transformer_encoder(multimodal_sequence)

    def fusion_attention(self, k, v,q):
        k=self.fc(k)
        v=self.fc(v)
        q=self.fc(q)
        attention_output, _ = self.attention(q, k, v) 
        return attention_output

