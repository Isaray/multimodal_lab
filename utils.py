from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.transforms import transforms
from transformers import BertTokenizer,RobertaTokenizer
import json
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torchvision
from data_preprocessing import *
import pandas as pd
seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
class MyDataset(Dataset):
    def __init__(self,args,df,pretrained_model,max_seq_length=128,n_classes=3):
        self.data = df
        self.args=args
        tokenizer=RobertaTokenizer.from_pretrained(pretrained_model)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transforms = transforms
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item=self.data.iloc[index]
        sentence = item["text"]
        # text_tokens = self.tokenizer(sentence, max_length=self.max_seq_length, truncation=True, padding='max_length',add_special_tokens=True) 
        text_tokens = self.tokenizer(sentence, return_tensors="pt", max_length=self.args.text_size,padding='max_length', truncation=True)
        text_tokens['input_ids'] = text_tokens['input_ids'].squeeze()
        text_tokens['attention_mask'] = text_tokens['attention_mask'].squeeze()
        # print(text_tokens)
        label = torch.tensor(item["tag"])

        image = Image.open(item["pic"]).convert("RGB")  
        pic_size = (self.args.pic_size, self.args.pic_size)
  
        transform = transforms.Compose([
        transforms.Resize(pic_size),
        transforms.RandomCrop(224),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])# refer to TomBert
        image_mat = transform(image).float()
        return item["guid"],text_tokens,image_mat,label


def load_data(args,pretrained_model="Robertabert-base"):
    if args.pretrained_model!=None:
      pretrained_model=args.pretrained_model
    img_size = (args.pic_size, args.pic_size)
    train_df=pd.read_csv("datasets/train_df.csv")
    dev_df=pd.read_csv("datasets/val_df.csv")
    test_df=pd.read_csv("datasets/test_df.csv")
    if args.train:
        train_set = MyDataset(args, train_df,pretrained_model)
        dev_set = MyDataset(args, dev_df,pretrained_model)
        return train_set, dev_set
    if args.test:
        test_set = MyDataset(args, test_df)
        return test_set
    
def save_data(file, predict_list):
    with open(file, 'w', encoding='utf-8') as f:
        f.write('guid,tag' + '\n')
        for i in range(len(predict_list)):
            f.write(predict_list[i]['guid'])
            f.write(',')
            f.write(predict_list[i]['tag'])
            f.write('\n')
    print('saved!')



