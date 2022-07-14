import time
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import load_data, save_data
from models.multi_model import  mBertModel,TomBertModel
from models.pic_model import PicClassifier,PicModel
from models.text_model import TextClassifier,TextModel
from tqdm import trange, tqdm
import pandas as pd
from transformers import BertTokenizer, BertModel,RobertaModel
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import accuracy_score




def train_one(args, train_dataloader, dev_dataloader):
    if args.train_kind=="text":
      model = TextClassifier(args).to(device=args.device)
    if args.train_kind=="pic":
      model = PicClassifier(args).to(device=args.device)
    if args.train_kind=="mbert":
      model = mBertModel(args).to(device=args.device)
    if args.train_kind=="tombert":
      model = TomBertModel(args).to(device=args.device)
    record=[]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    num_epochs = args.epoch
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")#可视化
        for step, batch in enumerate(epoch_iterator):
            ids, batch_text, batch_img, y = batch
            batch_text = batch_text.to(device=args.device)
            batch_img = batch_img.to(device=args.device)
            y = y.to(device=args.device)
            y_hat = model(batch_text=batch_text, batch_img=batch_img)
            loss=loss_func(y_hat, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0] #加上batchsize
            # print('step %d, loss %.4f, train acc %.3f' % (step, train_l_sum / n, train_acc_sum / n))
        print('epoch %d, loss %.4f, train acc %.3f' % (epoch, train_l_sum / n, train_acc_sum / n))
        val_loss,accuracy = evaluate_one(args, model, dev_dataloader, epoch)
        record.append([epoch,train_l_sum/n,val_loss,train_acc_sum/n,accuracy])

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.checkpoints_dir +args.train_kind+ str(best_accuracy) + '-checkpoint.pth')
    
    torch.save(model.state_dict(), args.checkpoints_dir +args.train_kind+ '-final_checkpoint.pth')
    df=pd.DataFrame(record)
    # print(df)
    df.columns=["epoch","train_loss","val_loss","train_acc","val_acc"]
    df.to_csv("datasets/new-{}-{}-{}.csv".format(args.train_kind,num_epochs,args.lr))


def evaluate_one(args, model, dev_dataloader, epoch=None):
    loss_func = nn.CrossEntropyLoss()
    dev_acc_sum_text, dev_acc_sum_img, dev_acc_sum, dev_acc_sum_ensemble, n = 0., 0., 0., 0., 0
    epoch_iterator = tqdm(dev_dataloader, desc="Evaluate")#可视化
    val_l_sum = 0
    for step, batch in enumerate(epoch_iterator):
        ids, batch_text, batch_img, y = batch
        batch_text = batch_text.to(device=args.device)
        batch_img = batch_img.to(device=args.device)
        y = y.to(device=args.device)
        y_hat = model(batch_text=batch_text, batch_img=batch_img)
        loss=loss_func(y_hat, y.long())
        dev_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()  
        n += y.shape[0]
        val_l_sum += loss.item()
    if epoch:
        print('epoch %d, dev acc %.4f(text), dev acc %.4f(image), dev acc %.4f(fusion),    dev acc %.4f(ensemble)'
              % (epoch, dev_acc_sum_text / n, dev_acc_sum_img / n, dev_acc_sum / n, dev_acc_sum_ensemble / n))
    else:
        print('dev acc %.4f(text), dev acc %.4f(image), dev acc %.4f(fusion),    dev acc %.4f(ensemble)'
              % (dev_acc_sum_text / n, dev_acc_sum_img / n, dev_acc_sum / n, dev_acc_sum_ensemble / n))
    return val_l_sum/n,dev_acc_sum / n


def test_one(args, test_dataloader, dev_dataloader):
    if args.train_kind=="text":
      model = TextClassifier(args).to(device=args.device)
    if args.train_kind=="pic":
      model = PicClassifier(args).to(device=args.device)
    if args.train_kind=="mbert":
      model = mBertModel(args).to(device=args.device)
    if args.train_kind=="tombert":
      model = TomBertModel(args).to(device=args.device)
    
    model.load_state_dict(torch.load(args.checkpoints_dir+ args.train_kind+ '-final_checkpoint.pth'))
    evaluate_one(args, model, dev_dataloader)
    
    predict_list = []
    for step, batch in enumerate(test_dataloader):
        ids, batch_text, batch_img, _ = batch
        batch_text = batch_text.to(device=args.device)
        batch_img = batch_img.to(device=args.device)
        y_hat = model(batch_text=batch_text, batch_img=batch_img)
        predict_y = y_hat.argmax(dim=1)  
         
        for i in range(len(ids)):
            item_id = ids[i]
            tag = int(predict_y[i])
            predict_dict = {
                'guid': item_id,
                'tag': tag,
            }
            predict_list.append(predict_dict)
    save_data(args.test_output_file, predict_list)


if __name__ == '__main__': 
    #local test
    class struct:
      def __init__(self):
        self.device="cuda"
        self.train=True
        self.test=True
        self.epoch=10
        self.batch_size=8
        self.pretrained_model="roberta-base"
        self.dropout=0.0
        self.lr=1e-5
        self.text_size=256
        self.pic_size=256
        self.checkpoints_dir="./output/"
        self.train_kind="mbert"
        self.test_output_file="./output/{}-result.txt".format(self.train_kind)

    arguments=struct()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-train_kind', '--train_kind',
    #                     type=str, default='mbert', help='model kind for train ')
    # parser.add_argument('-pretrained_model', '--pretrained_model',
    #                     type=str, default='roberta-base', help='pretrained_model')
    # parser.add_argument('--train', action='True', help='start training.')
    # parser.add_argument('--test', action='True', help='start testing.')
    # parser.add_argument("-lr", "--lr",
    #                     type=float, default=1e-5, help='learning rate')
    # parser.add_argument("-dropout", "--dropout",
    #                     type=float, default=0.0, help='dropout')
    # parser.add_argument("-epoch", "--epoch",
    #                     type=int, default=10, help='epoch')
    # parser.add_argument("-batch_size", "--batch_size",
    #                     type=int, default=5, help='batch size')
    # parser.add_argument('-checkpoints_dir', '--checkpoints_dir',
    #                     type=str, default='./checkpoint', help='checkpoint dir')
    # parser.add_argument('-test_output_file', '--test_output_file',
    #                     type=str, default='./test_with_label.txt', help='test output file')
    # parser.add_argument("--pic_size", "--pic_size",
    #                     type=int, default=256, help='pic size')
    # parser.add_argument("--text_size", "--text_size",
    #                     type=int, default=256, help='text size')

    # arguments = parser.parse_args()


    # cuda
    arguments.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:' + str(arguments.device))

    if arguments.train:
        train_set, dev_set = load_data(arguments)
        train_dataloader = DataLoader(train_set, shuffle=True, batch_size=arguments.batch_size)
        eval_dataloader = DataLoader(dev_set, shuffle=True, batch_size=arguments.batch_size)
        print('model training...')
        train_one(arguments, train_dataloader, eval_dataloader)
    if arguments.test:
        test_set, dev_set = load_data(arguments)
        test_dataloader = DataLoader(test_set, shuffle=False, batch_size=arguments.batch_size)
        eval_dataloader = DataLoader(dev_set, shuffle=False, batch_size=arguments.batch_size)
        print('model testing...')
        test_one(arguments, test_dataloader, eval_dataloader)




