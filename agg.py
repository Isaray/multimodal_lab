import pandas as pd
import numpy as np
df1=pd.read_csv("datasets/final-result1.txt")
df2=pd.read_csv("datasets/final-result2.txt")
df3=pd.read_csv("datasets/final-result3.txt")
data=[]
mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
rmapping = {0:'negative', 1:'neutral', 2:'positive'}


for i in range(len(df1)):
    l=[0 for i in range(3)]
    l[mapping[df1.iloc[i]['tag']]]=+0.745
    l[mapping[df2.iloc[i]['tag']]]=+0.749
    l[mapping[df3.iloc[i]['tag']]]=+0.727
    tag=np.argmax(l)
    data.append([df1.iloc[i]['guid'],rmapping[tag]])
res=pd.DataFrame(data,columns=['guid','tag'])
res.to_csv("test_with_label.txt",index=False)
# print(res)