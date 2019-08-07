'''Perform transforms on both PIL image and object boxes.

'''
import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.autograd import Variable

# GPU enable
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def resize(img, boxes, size, max_size=1000):
    '''Resize the input PIL image to the given size.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow,oh), Image.BILINEAR), \
           boxes*torch.Tensor([sw,sh,sw,sh])

def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    boxes -= torch.Tensor([x,y,x,y])
    boxes[:,0::2].clamp_(min=0, max=w-1)
    boxes[:,1::2].clamp_(min=0, max=h-1)
    return img, boxes


def compute_bbox_grid(feature, boxes, crop_size=16., img_size=128):
    '''
    args
        feature : (Tensor, sized [B,C,H,W]) feature to crop in crop_size
        bbox : (Tensor, sized [B,1,4]) bbox which is matched in image size
    return
        grid : F.affine_grid() according to the feature input size
    '''

    B = len(feature)
    rel_factor = float(crop_size)/img_size
    boxes = rel_factor*boxes
    bbox_w, bbox_h = boxes[:,0,2]-boxes[:,0,0], boxes[:,0,3]-boxes[:,0,1]
    bbox_cx, bbox_cy = (boxes[:,0,2]+boxes[:,0,0])/2., (boxes[:,0,3]+boxes[:,0,1])/2.
    w, h = crop_size, crop_size

    scale_x = w/bbox_w
    scale_y = h/bbox_h
    zero_x = Variable(torch.zeros(B)).to(device)
    zero_y = Variable(torch.zeros(B)).to(device)

    line0 = torch.stack((torch.zeros(B)        , torch.zeros(B)    , torch.zeros(B)  ))
    line0 = torch.stack((scale_x    , zero_x    , -(bbox_cx-w/2.)/(w/2.)  ))
    line1 = torch.stack((zero_y     , scale_y   , -(bbox_cy-h/2.)/(h/2.)  ))
    theta = torch.cat((line0,line1),0).transpose(0,1).view(-1,2,3)#.view(-1,6)
    grid = F.affine_grid(theta, feature.size())
    return grid

def center_crop(img, boxes, size):
    '''Crops the given PIL Image at the center.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size (tuple): desired output size of (w,h).

    Returns:
      img: (PIL.Image) center cropped image.
      boxes: (tensor) center cropped boxes.
    '''
    w, h = img.size
    ow, oh = size
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j+ow, i+oh))
    boxes -= torch.Tensor([j,i,j,i])
    boxes[:,0::2].clamp_(min=0, max=ow-1)
    boxes[:,1::2].clamp_(min=0, max=oh-1)
    return img, boxes

def random_flip(img, boxes):
    '''Randomly flip the given PIL Image.

    Args:
        img: (PIL Image) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
        img: (PIL.Image) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:,0] = xmin
        boxes[:,2] = xmax
    return img, boxes

def draw(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    img.show()


def test():
    img = Image.open('./image/000001.jpg')
    boxes = torch.Tensor([[48, 240, 195, 371], [8, 12, 352, 498]])
    img, boxes = random_crop(img, boxes)
    print(img.size)
    draw(img, boxes)

# test()
