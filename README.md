# MultiModal Lab

This experiment to build a multimodal sentiment analysis model, which analyzes image and text data to discern whether the combination of image and text is positive, negative or neutral.

## Framework

```
├─agg.py
├─data_preprocessing.py
├─run.py   #run this file to train/test the model
├─utils.py
├─models
|   ├─multi_model.py
|   ├─pic_model.py
|   ├─text_model.py
├─datasets
|    ├─test_df.csv
|    ├─test_without_label.txt
|    ├─train.txt  #ids and labels are here
|    ├─train_df.csv
|    ├─val_df.csv
|    ├─data #put pics in this dir
```

## Requirements

```
numpy==1.18.3
pandas==1.3.5
Pillow==9.2.0
scikit_learn==1.1.1
torch==1.11.0
torchvision==0.12.0
tqdm==4.64.0
transformers==4.17.0
```

```bash
# you can install them wiit:
pip install -r requirements.txt
```

## Usage

I used some pre-trained models and the transformer encoder and multihead attention modules that come with torch to build models similar to mBert and TomBert.

You can train the model with the following command line:

```
python run.py --train_kind mbert 
```



## Reference

[Adapting BERT for Target-Oriented Multimodal Sentiment Classification (ijcai.org)](https://www.ijcai.org/Proceedings/2019/0751.pdf)

[TomBERT/absa_data at master · jefferyYu/TomBERT (github.com)](https://github.com/jefferyYu/TomBERT)