import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def conv_base(channel_in=60,channel=60,layer=30,padding=1):
    conv_base = []
    for _ in range(layer):
        conv_base.append(nn.Sequential(nn.Conv2d(in_channels=channel_in,
                                                 out_channels=channel,
                                                 kernel_size=(3, 3),
                                                 padding=(padding, padding),
                                                 stride=(1, 1)),
                                       nn.BatchNorm2d(channel),
                                       nn.LeakyReLU(),
                                       ))

    return conv_base

class CapatchaModel (nn.Module):
    def __init__(self,label_size=(6,10),bn_size=128):
        super(CapatchaModel,self).__init__()

        #input = Batch x 1x128x128


        self.label_size=label_size
        self.channel = self.label_size[0]*self.label_size[1]

        self.bn_size =bn_size
        #self.bn = [nn.BatchNorm2d(self.channel) for cnt in range(5)]

        conv_base1 = conv_base(channel_in=3,channel=self.channel, layer=1)
        self.conv1 = nn.Sequential(*conv_base1)
        self.maxpool1   = nn.MaxPool2d((2,2))

        conv_base2 = conv_base(channel_in=self.channel,channel=self.channel, layer=10)
        self.conv2 = nn.Sequential(*conv_base2)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        conv_base3 = conv_base(channel_in=self.channel,channel=self.channel, layer=10)
        self.conv3 = nn.Sequential(*conv_base3)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        conv_base4 = conv_base(channel_in=self.channel,channel=self.channel, layer=10)
        self.conv4 = nn.Sequential(*conv_base4)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        conv_base5 = conv_base(channel_in=self.channel,channel=self.channel, layer=10)
        self.conv5 = nn.Sequential(*conv_base5)

        self.maxpool5 = nn.MaxPool2d((2, 2))

        #conv_base6= conv_base(channel=self.channel,layer=1)
        #self.conv6 = nn.Sequential(*conv_base6)
        #self.maxpool6 = nn.MaxPool2d((2, 2))

        self.linear = []
        for idx in range(3):
            in_channel = 352 if idx == 0 else self.channel
            self.linear.append( nn.Sequential( nn.Linear(in_channel,self.channel),
                                               nn.LeakyReLU(),
                                               #nn.Softmax(dim=-1),
                                               nn.Dropout(p=0.3))
               )
        self.linear =nn.Sequential(*self.linear)



    def forward(self,data):
  
        x1 = self.conv1(data)
        x1 = self.maxpool1(x1)
        x2 = self.conv2(    x1)
        x2 = self.maxpool2( x2)
        x3 = self.conv3(x2)
        x3 = self.maxpool3(x3)
        x4 = self.conv4(x3)
        x4 = self.maxpool4(x4)
        x5 = self.conv5(x4)

        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], x2.shape[2] * x2.shape[3]))
        x3 = torch.reshape(x3, (x3.shape[0], x3.shape[1], x3.shape[2] * x3.shape[3]))
        x4 = torch.reshape(x4, (x4.shape[0], x4.shape[1], x4.shape[2] * x4.shape[3]))
        x5 = torch.reshape(x5, (x5.shape[0], x5.shape[1], x5.shape[2] * x5.shape[3]))
        x6 = torch.concat((x2, x3, x4, x5), dim=-1)
        x = self.linear(x6)
        x = torch.norm(x,dim=-1)
        out = x.reshape(data.shape[0],self.label_size[0],self.label_size[1])
        out =F.softmax(out,dim=-1)
        return out



if __name__ =="__main__":
    data =torch.rand((5,3,64,64))
    model =CapatchaModel()
    #print(model)
    out =model(data)
    print(f'model output:{out.shape}')
