import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

PATH = os.getcwd()
data_path = PATH+'\images_restructure'



img_cols=184
img_rows=72
num_channel=1
num_epoch=24
num_classes = 27

img_data_list=[]
num_of_samples = 14308
label_index=0

labels = np.ones((num_of_samples,), dtype='int64')




def get_word_number_mapping(class_name):

    class_name = class_name.lower()
    print (class_name)
    word_number ={'crore':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'lakh':8,'nine':9,'ten':10,
     'thousand':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'hundred':18,
     'nineteen':19,'twenty':20,'thirty':21,'forty':22,'fifty':23,'sixty':24,'seventy':25,'ninety':26}


    res = word_number.get(class_name)

    print(res)
    return res


class_folders= os.listdir(data_path)

for cf in class_folders:
    data_path_cf = data_path+"\\"+cf
    os.chdir(data_path_cf)
    data_dir_list = [f for f in os.listdir(data_path_cf) if os.path.isfile(f)]

    #print(data_dir_list)

    for img in data_dir_list:

        print(cf)

        labels[label_index] = get_word_number_mapping(cf)
        print(labels[label_index])
        label_index += 1



        input_img = cv2.imread(data_path_cf + '\\' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img',input_img)
        cv2.waitKey(0)

        input_img_resize = cv2.resize(input_img, (img_cols, img_rows))

        cv2.imshow('img2',input_img_resize)
        cv2.waitKey(0)


        img_data_list.append(input_img_resize)




print(len(img_data_list))
print (labels.shape)









"""
for img in data_dir_list:

    class_name = img.split("_")[0]
    labels[label_index]  = get_word_number_mapping(class_name)
    label_index+=1

    input_img=cv2.imread(data_path + '/'+ img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img_resize = cv2.resize(input_img, (img_cols,img_rows))
    img_data_list.append(input_img_resize)


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        img_data = np.expand_dims(img_data, axis=1)
        print(img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=4)
        print(img_data.shape)

else:
    if K.image_dim_ordering() == 'th':
        img_data = np.rollaxis(img_data, 3, 1)
        print(img_data.shape)




# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

print

print(Y.shape)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# %%
# Defining the model
input_shape = img_data[0].shape

model = Sequential()

model.add(Convolution2D(32,3,3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])


model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

model.fit(X_train, y_train, batch_size=128, nb_epoch=num_epoch, verbose=1)
score = model.evaluate(X_test, y_test, batch_size=128)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# Testing a new image
test_image = cv2.imread('/datadrive/vijayendra/582.png')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image, (img_cols,img_rows))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print(test_image.shape)

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=3)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)

else:
    if K.image_dim_ordering() == 'th':
        test_image = np.rollaxis(test_image, 2, 0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)

# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))


"""