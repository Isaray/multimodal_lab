from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

class PicModel(nn.Module):
    def __init__(self, args):
        super(PicModel, self).__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)#IMAGENET1K_V2效果更好
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.encoder(x)
        return x

class PicClassifier(nn.Module):
    def __init__(self, args):
        super(PicClassifier, self).__init__()
        self.PicModel = PicModel(args)
        self.classifier_text = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 3),
        )
 
    def forward(self, batch_text=None, batch_img=None):
        pic_output = self.PicModel(batch_img)
        pic_output = self.classifier_text(pic_output)
        return pic_output