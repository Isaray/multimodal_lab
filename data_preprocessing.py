import json
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
import torch
import torchvision
import transformers
from torchvision import transforms
from PIL import Image

seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def get_train_val_df():
    train_total = pd.read_csv("datasets/train.txt")
    # test_index = list(pd.read_csv("datasets/test_without_label.txt")["guid"])
    mapping = {'negative': 0, 'neutral': 1, 'positive': 2, np.nan: 3}

    train_total=train_total.replace(mapping) 
    train_total = shuffle(train_total,random_state=seed).reset_index(drop=True)
    train_total_0=train_total[train_total.tag==0]
    train_total_1=train_total[train_total.tag==1]
    train_total_2=train_total[train_total.tag==2]
    #均匀抽样
    ratio=0.8
    train_0=train_total_0[:int(len(train_total_0)*ratio)]
    val_0=train_total_0[int(len(train_total_0)*ratio):]
    train_1=train_total_1[:int(len(train_total_1)*ratio)]
    val_1=train_total_1[int(len(train_total_1)*ratio):]
    train_2=train_total_2[:int(len(train_total_2)*ratio)]
    val_2=train_total_2[int(len(train_total_2)*ratio):]
    train_df=pd.concat([train_0,train_1,train_2])
    val_df=pd.concat([val_0,val_1,val_2])
    train_df["text"]=get_text_list(train_df)
    val_df["text"]=get_text_list(val_df)
    train_df["pic"]=get_pic_list(train_df)
    val_df["pic"]=get_pic_list(val_df)
    return train_df,val_df

def read_text(index):
    filepath="datasets/data/"
    encoding='gb18030'
    with open(filepath + str(index) + ".txt", encoding=encoding) as f:
        txt=f.read().rstrip("\n")#去掉最后一个\n
    return txt
def get_text_list(df):
    train_text=[]
    for i in range(len(df)):
        # print()
        txt=read_text(int(df.iloc[i].guid))
        train_text.append(txt)
    return train_text

def get_pic_list(df):
    train_text=[]
    for i in range(len(df)):
        # print()
        txt="datasets/data/"+str(df.iloc[i].guid)+".jpg"
        train_text.append(txt)
    return train_text

def get_test_df():
    test_df = pd.read_csv("datasets/test_without_label.txt")
    test_df["text"]=get_text_list(test_df)
    test_df["pic"]=get_pic_list(test_df)
    mapping = {'negative': 0, 'neutral': 1, 'positive': 2, np.nan: 3}
    test_df=test_df.replace(mapping) 
    # print(test_df)
    return test_df
# print(get_train_val_df())

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

# path="datasets/data/1.jpg"
# crop_size=224
# image = Image.open("datasets/data/1.jpg").convert("RGB")  # convert the image to three channels(RGB)
# transform = transforms.Compose([
#         transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))])

# image = image_process(path, transform)


# print(image.size())

def get_total_df():
    train_df,val_df=get_train_val_df()
    test_df=get_test_df()
    t=pd.concat([train_df,val_df,test_df])
    # print(t)
    return t
# get_total_df()

def save_preprocessed():
    train_df,val_df=get_train_val_df()
    train_df.to_csv("datasets/train_df.csv",index=False)
    val_df.to_csv("datasets/val_df.csv",index=False)
    test_df=get_test_df()
    test_df.to_csv("datasets/test_df.csv",index=False)

# save_preprocessed()








