# -*- coding: utf-8 -*-
"""
このコードは一部を除いて、MATHGRAM　by　k3nt0 (id:ket-30)さんの
以下のサイトのものを利用しています。
http://www.mathgram.xyz/entry/chainer/bake/part5
"""
from __future__ import print_function
from collections import defaultdict

from PIL import Image
from six.moves import range
import keras.backend as K

from keras.utils.generic_utils import Progbar
import numpy as np
import keras

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from keras.preprocessing import image
import sys
import cv2
import os


np.random.seed(1337)

K.set_image_data_format('channels_last')

#その１　------データセット作成------

#フォルダは整数で名前が付いています。
def getDataSet(img_rows,img_cols):
    #リストの作成
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(0,6):
        path = "./train_images/"
        if i == 0:
            #othersは600枚用意します。テスト用には60枚
            cutNum = 70
            cutNum2 = 60
        elif i == 1:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 70
            cutNum2 = 60
        elif i==2:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 70
            cutNum2 = 60
 
        elif i==3:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 70
            cutNum2 = 60
 
        elif i==4:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 70
            cutNum2 = 60
 
        else:
            #主要キャラたちは480枚ずつ。テスト用には40枚
            cutNum = 70
            cutNum2 = 60
        imgList = os.listdir(path+str(i))
        print(imgList)
        imgNum = len(imgList)
        for j in range(cutNum):
            #imgSrc = cv2.imread(path+str(i)+"/"+imgList[j])
            #target_size=(img_rows,img_cols)のサイズのimage
            img = image.load_img(path+str(i)+"/"+imgList[j], target_size=(img_rows,img_cols))
            imgSrc = image.img_to_array(img)
                        
            if imgSrc is None:continue
            if j < cutNum2:
                X_train.append(imgSrc)
                y_train.append(i)
            else:
                X_test.append(imgSrc)
                y_test.append(i)
    print(len(X_train),len(y_train),len(X_test),len(y_test))

    return X_train,y_train,X_test,y_test
