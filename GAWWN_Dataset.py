import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import os.path
import sys
import string
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import configparser
from transform import resize, random_flip, random_crop, center_crop, compute_bbox_grid


class GAWWN_Dataset(Dataset):
    def __init__(self, cfg, train, transform):
        '''
        args:
            cfg : path to the cfg file.
        return (get items):
            texts : (tuple, sized [B])
            bboxs : (tensor, sized [B,1,4])
            images : (tensor, sized [B,3,size,size])

        '''
        self.cfg = configparser.ConfigParser()
        self.cfg.read(cfg)
        self.figsize = int(self.cfg["Train"]["IMAGE_SIZE"])
        self.path_img_txt = self.cfg["Dataset"]["PATH_IMG_TXT"]
        self.path_img_folder = self.cfg["Dataset"]["PATH_IMG_FOLDER"]
        self.path_text_folder = self.cfg["Dataset"]["PATH_TEXT_FOLDER"]
        self.path_text_txt = self.cfg["Dataset"]["PATH_TEXT_TXT"]
        self.path_text_npy = self.cfg["Dataset"]["PATH_TEXT_NPY"]
        self.path_bbox_txt = self.cfg["Dataset"]["PATH_BBOX_TXT"]

        self.fnames = []
        self.bboxs = []
        self.texts = []
        self.texts_encoded = torch.from_numpy(np.load(self.path_text_npy))

        self.transform = transform
        self.train = train

        print("Loading cfg from :" ,cfg)
        with open(self.path_img_txt) as f_img:
            lines_img = f_img.readlines()
            self.num_imgs = len(lines_img)
        with open(self.path_bbox_txt) as f_bbox:
            lines_bbox = f_bbox.readlines()
        with open(self.path_text_txt) as f_text:
            lines_text = f_text.readlines()

        for idx, (line_img, line_bbox, line_text) in enumerate(zip(lines_img, lines_bbox, lines_text)):
            # img path loading
            splited = line_img.strip().split()
            img_idx = splited[0]
            img_path = splited[1]
            self.fnames.append(img_path)

            # bbox loading
            splited = line_bbox.strip().split()
            bbox_idx = splited[0]
            box = [float(splited[1]),float(splited[2]),float(splited[1])+float(splited[3]),float(splited[2])+float(splited[4])]
            self.bboxs.append(torch.Tensor(box).view(-1,4))

            # text loading
            splited = line_text.strip().split()
            text_idx = splited[0]
            text = line_text.strip().replace(text_idx+' ','')
            self.texts.append(text)

        print("Initializing succeed!")

    def __getitem__(self, index):
        img_path = self.fnames[index]
        bbox = self.bboxs[index]
        texts = self.texts[index]
        texts_encoded = self.texts_encoded[index]

        # loading img
        img = Image.open(os.path.join(self.path_img_folder, img_path))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.train:
            img, boxes = random_flip(img, bbox)
            img, boxes = random_crop(img, bbox)
            img, boxes = resize(img, bbox, (self.figsize,self.figsize))

        else:
            img, boxes = resize(img, bbox, (self.figsize,self.figsize))
            img, boxes = center_crop(img, bbox, (self.figsize,self.figsize))

        img = self.transform(img)
        return texts_encoded, img, boxes

    def __len__(self):
        return len(self.fnames)

def test():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    cfg_path = './config/GAWWN_v1.cfg'
    train_dataset = GAWWN_Dataset(cfg = cfg_path, train = True, transform = transform)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=4)
    train_iter = iter(train_loader)
    #print(len(train_loader.dataset))
    #print(len(train_loader))
    for epoch in range(1):
        texts, img, boxes = next(train_iter)
        print("image shape : ",img.shape)
        print("texts shape : ",texts.shape)
        print("boxes shape : ",boxes.shape)
        print("boxes : ",boxes)

        grid = torchvision.utils.make_grid(img, 1)
        if not os.path.exists("./test"):
            os.makedirs("./test")
        torchvision.utils.save_image(grid, './test/test.jpg')
        img = Image.open('./test/test.jpg')
        draw = ImageDraw.Draw(img)
        for i,(box,text) in enumerate(zip(boxes,texts)):
            draw.rectangle(list(box[0]), outline='red',width = 3)
        img.save('./test/test_bbox.jpg')

        #cropping testing


        img = Image.open('./test/test.jpg')
        img, _ = resize(img, boxes, (16,16))
        img_torch = transforms.ToTensor()(img).unsqueeze(0)
        img_torch = img_torch.repeat(10,1,1,1)
        boxes = boxes.repeat(10,1,1)
        print('img_torch : ',img_torch.shape)
        print('boxes : ',boxes.shape)
        grid = compute_bbox_grid(img_torch, boxes, crop_size=16., img_size=128)
        output = F.grid_sample(img_torch, grid)
        print('output : ',output.shape)
        new_img_torch = output[0]
        plt.imshow(new_img_torch.numpy().transpose(1,2,0))
        plt.savefig('./test/crop.jpg')

if __name__ == '__main__':
    test()
