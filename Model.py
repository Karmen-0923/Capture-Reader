import torch
import torch.nn as nn
import torch.nn.functional as F
class CaptchaReg(nn.Module):
    def __init__(self,channel=30):
        super(CaptchaReg,self).__init__()
        self.channel =channel
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.bn2 = nn.BatchNorm2d(self.channel)
        self.bn3 = nn.BatchNorm2d(self.channel)
        self.conv1 = nn.Sequential(     nn.Conv2d(in_channels=3,
                                        out_channels=self.channel,
                                        kernel_size=(3,3),
                                        padding=(1,1),
                                        stride=(1,1)),
                                        self.bn1,

                                        nn.LeakyReLU(negative_slope=0.2)     )
        self.maxpool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Sequential(     nn.Conv2d(in_channels=self.channel,
                                        out_channels=self.channel,
                                        kernel_size=(3,3),
                                        padding=(1,1),
                                        stride=(1,1)),
                                        self.bn2,

                                        nn.LeakyReLU(negative_slope=0.2)     )
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Sequential(     nn.Conv2d(in_channels=self.channel,
                                        out_channels=self.channel,
                                        kernel_size=(3,3),
                                        padding=(1,1),
                                        stride=(1,1)),
                                        self.bn3,

                                        nn.LeakyReLU(negative_slope=0.2)     )
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.linear1 =nn.Linear(256+64,160)
        self.linear2 = nn.Linear(160,12*12)
        self.linear3 = nn.Linear( 12 * 12 ,60)

    def forward(self,data):

        #data = Bx3x64x6x
        x1 = self.conv1(data)
        x1 = self.maxpool1(x1)
        x2 = self.conv2(x1)
        x2 = self.maxpool2(x2)
        x3 = self.conv3(x2)
        x3 = self.maxpool3(x3)
        x2 = torch.reshape(x2,(x2.shape[0],x2.shape[1],x2.shape[2]*x2.shape[3]))
        x3 = torch.reshape(x3, (x3.shape[0], x3.shape[1], x3.shape[2] * x3.shape[3]))
        #print(f'x2={x2.shape},x3={x3.shape}')
        #print(f'dim1 ={x2.shape[1]} ')
        x = torch.concat((x2,x3),dim=2)
        # x = Bxchannelx(32+16) x(32+16)= Bxchannelx48x48 = Bxchannelx2304
        x= F.leaky_relu( self.linear1(x))
        x= F.leaky_relu( self.linear2(x))
        x= F.leaky_relu( self.linear3(x))
        #Bxchannelx60
        x =torch.permute(x,(0,2,1)) #Bxchannelx60->Bx60xchannel
        x =torch.sum(x,dim=-1)

        x = x.reshape(x.shape[0],6,10)
        x = F.softmax(x, dim=-1)
        return x

if __name__ =="__main__":

    data = torch.rand ((5,3,64,64))
    model = CaptchaReg (channel=30)
    output =model(data)
    print(output.shape)
    #print(output)



