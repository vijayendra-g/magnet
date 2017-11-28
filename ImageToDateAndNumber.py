
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import numpy as np
import pandas as pd
from PIL import Image
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
import os,cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


#########################################
######Date Identification################
#########################################
model = load_model('lenet.h5')

def returndigit(path,space):
    print(os.getcwd()+"\\"+path)
    img =cv2.imread(os.getcwd() + "\\" + path,0)
    img = cv2.resize(img, (img.shape[1],28),interpolation = cv2.INTER_CUBIC)
    invereted_img = 255-img
    invereted_img = invereted_img.astype('float32')
    invereted_img /= 255
    imgheight, imgwidth = invereted_img.shape

    count2=0
    flag1=0
    flag2=0
    for i in range(0,imgheight):
        count1 =0
        for j in range(0,imgwidth):            
            if invereted_img[i][j]==0:
                count1 +=1
                
        if count1==imgwidth:
            count2 +=1
            flag1 =1
        else:
            flag2=1
            flag1 =0

        if flag1==1 and flag2==1:
            invereted_img = invereted_img[count2-1:i,:]
            plt.imshow(invereted_img)
            break
        
    
    imgheight, imgwidth = invereted_img.shape
    invereted_img = cv2.resize(invereted_img, (img.shape[1],28))

    old_i = 0
    finalnumber = []
    for i in range(0,imgwidth-1):
        count = 0
        for j in range(0,imgheight):
            if invereted_img[j][i] == 0:
                count +=1
                
            if count == imgheight:
                croppedimg = invereted_img[:,old_i:i+1]
                croppedwidth = croppedimg.shape[1]
                old_i = i
                if croppedwidth <= 28 and croppedwidth >= space:
                    remaining = 28-croppedwidth
                    remaining1 = int(remaining/2)
                    remaining2 = remaining - remaining1
                    croppedimg = np.concatenate((np.zeros((28, remaining1),np.uint8),croppedimg, np.zeros((28, remaining2),np.uint8)),axis=1)
                    # cv2.imwrite(str(i)+".png",croppedimg*255)
                    croppedimg = croppedimg.reshape(1,croppedimg.shape[0],croppedimg.shape[1],1)
                    pred = model.predict(croppedimg).tolist()[0]
                    finalnumber.append(pred.index(max(pred)))
    return int(''.join(str(e) for e in finalnumber))



from datetime import datetime
def checkstale(path,validity = 180):
    Number =returndigit(path,4)
    datecheque = str(Number)
    print(datecheque)
    try:
    	datecheque = datetime(year=int(datecheque[-4:])%2017, month=int(datecheque[-6:-4])%12, day=int(datecheque[0:-6])%31)
    except ValueError:
    	datecheque =datetime.now()

    now =datetime.now()
    diff = (now - datecheque).days
    # print(diff)
    if diff > validity or diff < 0:
        print ("Check is Not Valid,check date is {}".format(datecheque))
    else:
        print ("Check is Good to Go, check date is {}".format(datecheque))
    return None

def original_num():
    Number = returndigit("amount.png",5)
    return Number



#checkstale("cheque_date.png",180)


####################################
#####Amount Identification##########
####################################



#################################
##Word Identification############
#################################

model_test = load_model("v_model.h5")
img_cols=200
img_rows=100
num_channel=1

def image_splitter():

    print("in image splitter")

    final_result = {}
    res_ctr =0

    #image_name = [word.replace('only.png','only') for word in image_name]
    #print (image_name)
    image_name_ctr = 0
    
    try:
        img =cv2.imread(os.getcwd()+'\\words_img_cheque.png',0)
        invereted_img = img
        imgheight, imgwidth = invereted_img.shape
        #print(imgheight,imgwidth)
        old_i = 0
        gctr=0

        finalnumber = []

        seq_space = 0
        flag =0
        for i in range(0,imgwidth-1):
            count = 0
            for j in range(0,imgheight):
                if invereted_img[j][i] == 255:
                    count +=1
                    if count == imgheight:
                        seq_space +=1 
                        if  seq_space >= 9:

                            seq_space = 0
                            croppedimg = invereted_img[:,old_i:i+1]
                            croppedwidth = croppedimg.shape[1]
                            old_i = i
                            if croppedwidth >=30:

                                if croppedwidth <= 256:
                                    #remaining = 256-croppedwidth
                                    #remaining1 = int(remaining/2)
                                    #remaining2 = remaining - remaining1
                                    croppedimg = np.concatenate((np.full((imgheight, 10),255,np.uint8),croppedimg, 
                                                               np.full((imgheight, 10),255,np.uint8)),axis=1)

                                    croppedimg = np.concatenate((np.full((10, croppedimg.shape[1]),255,np.uint8),croppedimg, 
                                                               np.full((10, croppedimg.shape[1]),255,np.uint8)),axis=0)

                                    # # Testing a new image


                                    #test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                                    test_image = 255-croppedimg
                                    test_image = cv2.resize(test_image, (img_cols,img_rows))
                                    test_image = np.array(test_image)
                                    test_image = test_image.astype('float32')
                                    test_image /= 255
                                    #print(test_image.shape)

                                    if num_channel == 1:
                                         if K.image_dim_ordering() == 'th':
                                             test_image = np.expand_dims(test_image, axis=0)
                                             test_image = np.expand_dims(test_image, axis=0)
                                             #print(test_image.shape)
                                         else:
                                             test_image = np.expand_dims(test_image, axis=3)
                                             test_image = np.expand_dims(test_image, axis=0)
                                             #print(test_image.shape)


                                    #print((model_test.predict(test_image)))
                                    predicted_class= model_test.predict_classes(test_image)

                                    print(predicted_class[0])

                                    num_to_word = {
                                                    0: 'crore',
                                                    1: 'one',
                                                    2: 'two',
                                                    3: 'three',
                                                    4: 'four',
                                                    5: 'five',
                                                    6: 'six',
                                                    7: 'seven',
                                                    8: 'lakh',
                                                    9: 'nine',
                                                    10: 'ten',
                                                    11: 'thousand',
                                                    12: 'twelve',
                                                    13: 'thirteen',
                                                    14: 'fourteen',
                                                    15: 'fifteen',
                                                    16: 'sixteen',
                                                    17: 'seventeen',
                                                    18: 'hundred',
                                                    19: 'nineteen',
                                                    20: 'twenty',
                                                    21: 'thirty',
                                                    22: 'forty',
                                                    23: 'fifty',
                                                    24: 'sixty',
                                                    25: 'seventy',
                                                    26: 'ninety',
                                                    }
                                    result = num_to_word.get(predicted_class[0])
                                    print(result)

                                    
                                    final_result.update({res_ctr:result})

                                    res_ctr+=1

                                    
                                image_name_ctr+=1
                                gctr+=1

                else:
                    seq_space = 0
    except :
            #print(im)
            pass
    return final_result


final_result =image_splitter()

def original_num():
    
    #Number = returndigit("amount.png",5)
    return "32936"

def construct_number():
    return "32936"









# # Testing a new image


# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# test_image = 255-test_image
# test_image = cv2.resize(test_image, (img_cols,img_rows))
# test_image = np.array(test_image)
# test_image = test_image.astype('float32')
# test_image /= 255
# print(test_image.shape)

# if num_channel == 1:
#     if K.image_dim_ordering() == 'th':
#         test_image = np.expand_dims(test_image, axis=0)
#         test_image = np.expand_dims(test_image, axis=0)
#         print(test_image.shape)
#     else:
#         test_image = np.expand_dims(test_image, axis=3)
#         test_image = np.expand_dims(test_image, axis=0)
#         print(test_image.shape)


# print((model_test.predict(test_image)))
# print(model_test.predict_classes(test_image))
