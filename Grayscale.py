import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os
import glob
import tqdm
import random


def splittraintest():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    filename = os.listdir(os.path.join(DATA_DIR, 'Captchaset'))
    random.shuffle(filename)
    sizeofdataset =len(filename)
    train_set_size = int(0.75*sizeofdataset)


    train_set = filename[0:train_set_size]
    valid_set = filename[train_set_size+1:train_set_size+int(0.05*sizeofdataset)+1]
    test_set  = filename[train_set_size+int(0.1*sizeofdataset)+2:]

    print(f'total:{sizeofdataset},train:{len(train_set)},val:{len(valid_set)},test:{len(test_set)}')

    for i in range(len(train_set)):
        open(os.path.join(DATA_DIR,'train_data.txt'),'a').write(str(train_set[i])+'\n')
    for i in range(len(valid_set)):
        open(os.path.join(DATA_DIR,'valid_data.txt'),'a').write(str(valid_set[i])+'\n')
    for i in range(len(test_set)):
        open(os.path.join(DATA_DIR,'test_data.txt'),'a').write(str(test_set[i])+'\n')

def texttoTensor (label):
    labeltensor = torch.ones((6,10))/6
    """"
    0,0,0,....
    0,0,0,....
    0,0,0,....
    0,0,0,....
    1,1,1,1,1,1,1,1,1,1....
    """
    for i in range(6):

        #print(f'label[i]={label[i]}')
        labelconvtor = int(label[i])

        #print(f'label[i]={label[i]},labcontor = {labelconvtor}')
        labeltensor[i,labelconvtor] = 1
    #print(f'labeltensor ={labeltensor.unsqueeze(0).shape}')
    return labeltensor.reshape(1,labeltensor.shape[0],labeltensor.shape[1])


def load_data_file(partition='train'):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR,f'{partition}_data.txt')):
        print(f'processing data splitting....')
        splittraintest()
    txt_file = open(os.path.join(DATA_DIR,f'{partition}_data.txt'))
    #print(txt_file)

    ImageSet = torch.tensor([])
    labelSet = torch.tensor([])
    for file in txt_file:


        image = cv2.imread(os.path.join(DATA_DIR,'Captchaset',file[:-1]))
        image = cv2.resize(image,(64,64))
        image = cv2.medianBlur(image,3)
        #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # np.array
        image_gray = torch.tensor(image) # from np.arry to tensor
        #print(image_gray.shape)
        image_gray = image_gray.unsqueeze(0)  # add channel dimension 1306*256,256,3 ->802,256,256,3 ->no of image,width,height,channel
        #print(image_gray.shape)
        ImageSet = torch.concat([ImageSet,image_gray],dim=0)
        filename = file[:6] #xxxxx.png\n
        filename = texttoTensor(filename)

        labelSet = torch.concat([labelSet,filename])



    return ImageSet.permute((0,3,1,2)),labelSet

class CaptchaDataset(Dataset):
    def __init__(self,partition='train'):
        self.partition=partition #'train, test,val
        self.Image,self.label = load_data_file(self.partition)

    def __getitem__(self,item):
        return self.Image[item],self.label[item]

    def __len__(self):
        return self.Image.shape[0]







if __name__ == '__main__':
    #Imageset,labelset =load_data_file()
    #print(labelset.shape)
    #splittraintest()
    capttcha =CaptchaDataset(partition='train')
    traindataloader = DataLoader(dataset=CaptchaDataset(partition='test'),batch_size=5,shuffle=True)
    for idx , (image,label) in enumerate(tqdm.tqdm(traindataloader)):
        print(idx)
        #print(image.shape)
        print(label.shape)
    print()

