# Learning What and Where to Draw

### TODO
Idx | Tasks | Done
:------|:------|:----
1 | CUB 11 Dataset setting | July 29.2019
2 | Text Enbedding using Google Bert | July 29.2019
3 | Dataset.py Transforms.py combine | July 30.2019
4 | Train.py model.py C&debug | July 31.2019
5 | Run the test script (only the vanilla GAN) successfully | Aug 02.2019
6 | Run the GAWWN script successfully |

---

### Usage
- Get dataset here : (CUB) http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Bert_preprocessing.py to create the Bert text encoding
- modify the config.cfg
- python3 GAWWN_train_img_only.py for vanilla GAN training
- python3 GAWWN_train.py for GAWWN training (Not yet completed)

---

### CUB results

Types     | images
----------|:------:
vanilla GAN v1 basic structure from mnist test (epoch 100) | ![01](/images/GAN_v1/100.jpg)
vanilla GAN v1 basic structure from mnist test (epoch 200) | ![01](/images/GAN_v1/200.jpg)
vanilla GAN v2 add LeakyReLU, BN, deep structure (epoch 100, 64 * 64) | ![01](/images/GAN_v2_64/100_0.jpg)
vanilla GAN v2 add LeakyReLU, BN, deep structure (epoch 200, 64 * 64) | ![01](/images/GAN_v2_64/200_0.jpg)
vanilla GAN v2 add LeakyReLU, BN, deep structure (epoch 300, 64 * 64) | ![01](/images/GAN_v2_64/300_0.jpg)
vanilla GAN v2 add LeakyReLU, BN, deep structure (epoch 300, 128 * 128) | ![01](/images/GAN_v2_128/100_128.jpg)
vanilla GAN v2 add LeakyReLU, BN, deep structure (epoch 300, 128 * 128) | ![01](/images/GAN_v2_128/200_128.jpg)
vanilla GAN v2 add LeakyReLU, BN, deep structure (epoch 300, 128 * 128) | ![01](/images/GAN_v2_128/300_128.jpg)
vanilla GAN v3 add balanced G&D structure (epoch 300, 128 * 128) | ![01](/images/GAN_v3/300_128.jpg)
vanilla GAN v3 add balanced G&D structure and soft BCELoss (epoch 300, 128 * 128) | ![01](/images/GAN_v3_soft/soft_300.jpg)
