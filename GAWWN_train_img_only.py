import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import skimage.io
import skimage
import os
import time
import pandas as pd
import random
import pickle
import GAWWN_model_img_only
from GAWWN_Dataset import GAWWN_Dataset
import configparser
import argparse

parser = argparse.ArgumentParser(description='PyTorch GAWWN Training')
parser.add_argument('--ckpt', default="N", type=str, help='save img name')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
#parser.add_argument('--ckpt', type=str, default = 'ckpt',help='the name of ckpt (default : ckpt.pth)')
args = parser.parse_args()

random.seed(312)
torch.manual_seed(312)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def main():
    cfg_path = './config/GAWWN_v1.cfg'
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    # parameters
    learning_rate = float(cfg["Train"]["LEARNING_RATE"])
    num_epochs = 300
    batch_size = 32
    input_dim = int(cfg["Train"]["DIM_INPUT"])

    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./save_imgs"):
        os.makedirs("./save_imgs")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    # load my Dataset
    cfg_path = './config/GAWWN_v1.cfg'
    train_dataset = GAWWN_Dataset(cfg = cfg_path, train = True, transform = transform)
    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True,num_workers=4)

    print('the dataset has %d size.' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))

    # fixed input for model eval
    with torch.no_grad():
        fixed_inputs = Variable(torch.randn(9,256, 1, 1))
        fixed_text_feature = Variable(train_dataset.texts_encoded[:9])
        fixed_bboxs = Variable(torch.stack(train_dataset.bboxs[:9]))

    # models setting
    G_model = GAWWN_model_img_only.Generator(cfg)
    D_model = GAWWN_model_img_only.Discriminator(cfg)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        fixed_inputs = fixed_inputs.to(device)
        fixed_text_feature = fixed_text_feature.to(device)
        fixed_bboxs = fixed_bboxs.to(device)
        G_model ,D_model = G_model.to(device) ,D_model.to(device)

    # setup optimizer
    G_optimizer = optim.Adam(G_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    D_loss_list = []
    G_loss_list = []
    D_real_acc_list = []
    D_fake_acc_list = []

    print("Starting training...")

    for epoch in range(num_epochs):
        print("Epoch:", epoch+1)
        G_model.train()
        D_model.train()
        epoch_D_loss = 0.0
        epoch_G_loss = 0.0
        D_real_total_acc = 0.0
        D_fake_total_acc = 0.0

        if (epoch) in [50,100]:
            G_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        for idx, (texts_feature, imgs, boxes) in enumerate(train_loader):
            batch_size = len(imgs)
            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)
            texts_feature = Variable(texts_feature).to(device)
            imgs = Variable(imgs).to(device)
            boxes = Variable(boxes).to(device)

            real_labels = Variable(real_labels).to(device)
            fake_labels = Variable(fake_labels).to(device)

            # train the Generator
            G_model.zero_grad()
            z = torch.randn(batch_size, 256, 1, 1)
            z = Variable(z).to(device)


            fake_img = G_model(z,texts_feature,boxes)
            outputs = D_model(fake_img,texts_feature,boxes)

            G_loss = criterion(outputs, real_labels)

            epoch_G_loss += G_loss.item()
            G_loss.backward()
            G_optimizer.step()

            # train the Discriminator
            # BCE_Loss(x, y) = - y * log(D(x)) - (1-y) * log(1 - D(x))
            # real images , real_labels == 1

            D_model.zero_grad()
            if epoch < 100:
                noise = Variable(imgs.data.new(imgs.size()).normal_(0., 0.5))
                outputs = D_model(imgs,texts_feature,boxes)
            else:
                outputs = D_model(imgs,texts_feature,boxes)
            D_real_loss = criterion(outputs, real_labels)
            D_real_acc = np.mean((outputs > 0.5).cpu().data.numpy())

            # fake images
            # First term of the loss is always zero since fake_labels == 0
            # we don't want to colculate the G gradient
            outputs = D_model(fake_img.detach(),texts_feature,boxes)

            D_fake_loss = criterion(outputs, fake_labels)
            D_fake_acc = np.mean((outputs < 0.5).cpu().data.numpy())
            D_loss =  (D_real_loss + D_fake_loss) / 2.

            D_loss.backward()
            D_optimizer.step()

            D_real_total_acc += D_real_acc
            D_fake_total_acc += D_fake_acc
            epoch_D_loss += D_loss.item()
            print('Eph[%d/%d]| Itr[%d/%d]| Gloss %.4f Dloss %.4f| LR = %.4f| F_acc %.4f R_acc %.4f'
            %(epoch, num_epochs, idx, len(train_loader), epoch_G_loss/(idx+1),
                epoch_D_loss/(idx+1), learning_rate,D_fake_total_acc/(idx+1),D_real_total_acc/(idx+1)))


        if (epoch) % 30 == 0:
            save_checkpoint('./save/GAWWN-G-%03i.pth' % (epoch) , G_model, G_optimizer)
            save_checkpoint('./save/GAWWN-D-%03i.pth' % (epoch) , D_model, D_optimizer)

        # save loss data
        D_loss_list.append(epoch_D_loss/len(train_loader.dataset))
        G_loss_list.append(epoch_G_loss/len(train_loader.dataset))
        D_real_acc_list.append(D_real_total_acc/len(train_loader))
        D_fake_acc_list.append(D_fake_total_acc/len(train_loader))

        # testing
        G_model.eval()
        test_output = G_model(fixed_inputs,fixed_text_feature,fixed_bboxs)
        torchvision.utils.save_image(test_output.cpu().data,
                                './save_imgs/%s-%03d.jpg' %(args.ckpt,epoch+1), nrow=3)
        # epoch done
        print('-'*88)
    #
    with open('./logfile/D_loss.pkl', 'wb') as f:
        pickle.dump(D_loss_list, f)
    with open('./logfile/G_loss.pkl', 'wb') as f:
        pickle.dump(G_loss_list, f)
    with open('./logfile/D_real_acc.pkl', 'wb') as f:
        pickle.dump(D_real_acc_list, f)
    with open('./logfile/D_fake_acc.pkl', 'wb') as f:
        pickle.dump(D_fake_acc_list, f)

    # shuffle
if __name__ == '__main__':
    main()
