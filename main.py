import torch
import torch.nn as nn
from  Grayscale import CaptchaDataset
from torch.utils.data import  DataLoader
import  tqdm
from Model import CaptchaReg
from util import loss_cal
import visdom
from torch.optim.lr_scheduler import CosineAnnealingLR

vis =visdom.Visdom(env='Captcha')
Captcha =CaptchaDataset(partition='train')
traindataloader = DataLoader(dataset=Captcha,batch_size=5)
Captcha_val =CaptchaDataset(partition='valid')
validdataloader = DataLoader(dataset=Captcha_val,batch_size=1)

Captcha_test =CaptchaDataset(partition='test')
testdataloader  = DataLoader(dataset=Captcha_test,batch_size=1)
model = CaptchaReg(channel=60)
opt = torch.optim.Adam(model.parameters(),lr=0.0001)

epochs =200
schedule = CosineAnnealingLR(optimizer=opt,T_max=epochs)
loss_sum =0
count = 0
epoch_list = []
loss_list =[]
for epoch in range(epochs):
        #print(f'Now Run epoch {epoch}...')
        model.train()
        schedule.step()
        for idx ,(image,label) in enumerate( tqdm.tqdm(traindataloader)):
                #print(idx)
                #print(image.shape) #batch,channel,width,height
                #print(label.shape) #batch, char,charvalue


                opt.zero_grad()
                logit = model(image)
                loss = loss_cal(logit,label,smoothing=True)

                loss.backward()
                opt.step()
                loss_sum = loss_sum +loss.item()
                count = count+1
                if idx %49 ==0 :
                        print(f'ita :{idx},loss:{loss_sum/count}')

        print(f'epoch :{epoch},loss:{loss_sum/count},lr:{schedule.get_last_lr()}')
        epoch_list.append(epoch)
        loss_list.append(loss_sum/count)
        vis.scatter(torch.tensor([epoch]), torch.tensor([loss_sum/count]), win='Test Loss', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'})
        model.eval()
        for idx, (image, label) in enumerate(tqdm.tqdm(validdataloader)):
                logit = model(image)  # batch ,6,10
                logit_max = torch.max(logit, dim=-1)[1]  # batch ,6,1

                # print(logit_max) #thank you
                label_max = torch.max(label, dim=-1)[1]
                if idx%49==0:
                        print(f'pred:{logit_max[0]} , ture:{label_max[0]},mean:{torch.sum (torch.where(label_max == label_max)[0])/6 }')

model.eval()
print('Enter Eval state....')
for idx,(image,label) in enumerate(tqdm.tqdm(testdataloader)):
        logit = model(image) #batch ,6,10
        logit_max = torch.max(logit,dim=-1)[1] #batch ,6,1

        #print(logit_max)
        label_max = torch.max(label,dim=-1)[1]
        print(f'pred:{logit_max} , ture:{label_max}')






