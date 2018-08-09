import ssl
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import backend as k
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 移除ssl校验
ssl._create_default_https_context = ssl._create_unverified_context

# load mnist data
(X_train,Y_train),(X_test,Y_test) = mnist.load_data();

fig = plt.figure()

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(X_train[i],cmap='gray',interpolation='none')
    plt.title("Digit: ()".format(Y_train[i]))
    plt.xticks([])
    plt.yticks([])
fig

print("X_train shape",X_train.shape)
print("Y_tain shape",Y_train.shape)
print("X_test shape",X_test.shape)
print("Y_test shape",Y_test.shape)

img_rows,img_cols = 28,28
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows,img_cols)
    X_test = X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
    input_shape = (1, img_rows,img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0],img_rows, img_cols,1)
    X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols,1)
    input_shape = (img_rows, img_cols,1)

# more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test/= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(np.unique(Y_train, return_counts=True))


#set number of categories#set num
num_category = 10

Y_train= keras.utils.to_categorical(Y_train, num_category)
Y_test = keras.utils.to_categorical(Y_test, num_category)
Y_train[0]

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape = input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_category,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

batch_size = 128
num_epoch = 10

model_log = model.fit(X_train,Y_train,batch_size = batch_size,epochs=num_epoch,verbose=1,validation_data=(X_test,Y_test))

score = model.evaluate(X_test,Y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accurary')
plt.ylabel('accurary')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig

#Save the model
# serialize model to JSON
model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
model.save_weights("model_digit.h5")
print("Saved model to disk")
