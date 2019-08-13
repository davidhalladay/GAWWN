'''
Author : David Fan
E-mail : christine5200312@gmail.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transform import resize, random_flip, random_crop, center_crop, compute_bbox_grid

class Generator(nn.Module):
    def __init__(self, cfg, feature_size = 1024):
        super(Generator, self).__init__()

        self.img_size = int(cfg["Train"]["IMAGE_SIZE"])
        self.text_feature_size = int(cfg["Model"]["DIM_TEXT_FEATURE"])
        self.crop_size = int(cfg["Model"]["CROP_SIZE"])

        def upsample(in_feat, out_feat, normalize = True , _kernel_size=4, _stride = 2 , _padding = 1):
            layers = [nn.ConvTranspose2d(in_feat, out_feat,kernel_size=_kernel_size, stride = _stride, padding = _padding, bias=False)]
            if normalize :
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        def downsample(in_feat, out_feat, normalize = True , _stride = 2 , _padding = 1):
            layers = [nn.Conv2d(in_feat, out_feat,kernel_size=4, stride = _stride, padding = _padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.text_conv = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride = 1, padding = 0, bias=False)
        )

        self.encoder = nn.Sequential(
            # state : (B,768,16,16)
            *downsample(768, feature_size),
            # state : (B,1024,8,8)
            *downsample(feature_size, 2*feature_size),
            # state : (B,2048,4,4)
            *downsample(2*feature_size, feature_size),
            # state : (B,2048,2,2)
            *downsample(feature_size, feature_size)
        )

        self.deconv_local = nn.Sequential(
            # state : (B,2048,1,1)
            *upsample(2048, 1024, _stride = 1, _padding = 0),
            # state: (B,1024,4,4)
            *upsample(1024, 1024),
            # state: (B,512,8,8)
            *upsample(1024, 512),
            # state: (B,128,16,16)
            # nn.ConvTranspose2d(self.img_size , 3, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.Tanh()
        )

        #self.fc = nn.Linear(128, self.img_size * 4)
        self.deconv_global = nn.Sequential(
            # state : (B,1024,1,1)
            #nn.ConvTranspose2d(128, 128,kernel_size=3, stride = 1, padding = 1, bias=False),
            *upsample(1024, self.img_size * 8, normalize = False , _kernel_size=4, _stride = 1 , _padding = 0),
            # state: (B,1024,4,4)

            *upsample(self.img_size * 8, self.img_size * 4),

            *upsample(self.img_size * 4, self.img_size * 2),
            # state: (B,512,8,8)
            *upsample(self.img_size * 2, self.img_size * 1),
            # state: (B,128,16,16)
            # nn.ConvTranspose2d(self.img_size , 3, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.Tanh()
        )

        self.deconv_comb = nn.Sequential(
            # state : (B,1024,16,16)
            #*upsample(self.img_size * 1, self.img_size ),
            # state: (B,512,32,32)
            # state: (B,512,64,64)
            nn.ConvTranspose2d(self.img_size , 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, texts_feature, boxes):
        '''
        args
            x : (Tensor, sized [B, input_dim, 1, 1])
            texts_feature : (Tensor, sized [B, 1, 768])
        '''
        Batch_size = len(x)
        ####################################################
        # STN part
        #texts_cube = texts_feature.repeat(1,1,16*16).view(-1,16,16,768).transpose(1,3).transpose(2,3)
        #grid = compute_bbox_grid(texts_cube, boxes, crop_size=16., img_size=128)
        #crop_box = F.grid_sample(texts_cube, grid)
        #input_feature = self.encoder(crop_box)

        input_comb_noise = torch.cat((x,texts_feature.view(Batch_size,768,1,1)),dim = 1)

        ####################################################
        # Local pathway
        #local_cube = self.deconv_local(input_comb_noise)
        #local_grid = compute_bbox_grid(local_cube, boxes, crop_size=16., img_size=128)
        #local_crop_cube = F.grid_sample(local_cube, local_grid)

        ####################################################
        # Global pathway
        #x = self.fc(x.view(Batch_size,-1)).view(Batch_size,self.img_size,2,2)
        global_cube = self.deconv_global(input_comb_noise)

        ####################################################
        # Combination
        #com_feature = torch.cat((global_cube,local_crop_cube),dim = 1)
        output = self.deconv_comb(global_cube)
        #output = output/2.0+0.5

        return output



class Discriminator(nn.Module):
    def __init__(self, cfg, feature_size = 1024):
        super(Discriminator, self).__init__()

        self.img_size = int(cfg["Train"]["IMAGE_SIZE"])
        self.text_feature_size = int(cfg["Model"]["DIM_TEXT_FEATURE"])
        self.crop_size = int(cfg["Model"]["CROP_SIZE"])

        def upsample(in_feat, out_feat, normalize = True , _stride = 2 , _padding = 1):
            layers = [nn.ConvTranspose2d(in_feat, out_feat,kernel_size=4, stride = _stride, padding = _padding, bias=False)]
            if normalize :
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(inplace = True))
            return layers

        def downsample(in_feat, out_feat, normalize = True , _kernel_size = 4, _stride = 2 , _padding = 1):
            layers = [nn.Conv2d(in_feat, out_feat,kernel_size=_kernel_size, stride = _stride, padding = _padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.text_conv = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride = 1, padding = 0, bias=False)
        )

        self.conv_local = nn.Sequential(
            # state: (B,3,128,128)
            *downsample(3, self.img_size),
            # state: (B,figsize,64,64)
            *downsample(self.img_size, self.img_size * 2),
            # state: (B,figsize*2,32,32)
            *downsample(self.img_size * 2, self.img_size * 4),
            # state: (B,figsize*4,16,16)
            #nn.Conv2d(figsize * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            #nn.Sigmoid()
        )

        self.conv_comb = nn.Sequential(
            # state: (B,768,16,16)
            *downsample(768, 512),
            # state: (B,figsize,8,8)
            *downsample(512, 256),
            # state: (B,figsize*2,4,4)
            *downsample(256, 128),
            # state: (B,figsize*4,2,2)
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Sigmoid()
        )

        self.conv_global = nn.Sequential(
            # state: (B,3,128,128)
            *downsample(3, self.img_size),
            # state: (B,4,64,64)
            *downsample(self.img_size, self.img_size * 2),
            # state: (B,16,8,8)
            *downsample(self.img_size * 2, self.img_size * 4),
            # state: (B,16,8,8)
            *downsample(self.img_size * 4,self.img_size * 8),


            # state: (B,48,4,4)
            nn.Conv2d(self.img_size * 8, 768, kernel_size=4, stride=1, padding=0, bias=False),
            #nn.Sigmoid()
        )

        self.out_model = nn.Sequential(
            # state: (B,256,1,1)

            nn.Conv2d(768, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, texts_feature, boxes, noise = None, if_noise = False):
        '''
        args
            x : (Tensor, sized [B, 3, 128, 128])
            texts_feature : (Tensor, sized [B, 1, 768])
        '''
        #texts_feature = self.text_conv(texts_feature.view(-1,768,1,1))

        Batch_size = len(x)
        if if_noise : input = x + noise
        else : input = x

        ####################################################
        # STN part
        #texts_cube = texts_feature.view(Batch_size,1,-1).repeat(1,1,16*16).view(-1,16,16,256).transpose(1,3).transpose(2,3)

        ####################################################
        # Local pathway
        #local_cube = self.conv_local(x)
        #local_camb_cube = torch.cat((local_cube,texts_cube),dim = 1)
        #grid = compute_bbox_grid(local_camb_cube, boxes, crop_size=16., img_size=128)
        #local_crop_box = F.grid_sample(local_camb_cube, grid)
        #local_feature = self.conv_comb(local_crop_box)

        ####################################################
        # Global pathway
        #print(x.shape)
        global_feature = self.conv_global(input)#.view(Batch_size,-1,1,1)
        global_add = global_feature + texts_feature.view(Batch_size,768,1,1)#torch.cat((global_feature,texts_feature.view(Batch_size,768,1,1)),dim = 1)
        #print(global_feature.shape)
        #comb_feature = local_feature.view(Batch_size,-1,1,1) + global_add

        output = self.out_model(global_add)
        output = output.view(-1, 1).squeeze(1)

        return output
