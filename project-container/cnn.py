
# coding: utf-8

# # CNN

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly as plty
import sklearn as skl

import matplotlib.mlab as mlabQ
import seaborn as sns

from copy import copy as copy
from pprint import pprint

import sys
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[2]:


import tensorflow as tf
import keras


# In[19]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


# In[20]:


ds_path = os.path.abspath("./") + "/cifar-10-batches-py/"

batch_container = [ None for _ in range(5)]

for ii in os.listdir(ds_path):
    if "data" in ii:
        current_id = int(ii[-1]) - 1
        batch_container[current_id] = unpickle(ds_path + ii)
        
        dummy = batch_container[current_id]
        kkeys = list(dummy.keys())

        for kk in kkeys:
            dummy[repr(kk)[2:-1]] = dummy.pop(kk)      


# In[21]:


keys_ids = ['batch_label', 'labels', 'data', 'filenames']


# In[22]:


for ii in range(5):
    print(ii + 1)
    print(repr(batch_container[ii][keys_ids[0]])[2:-1])
    for keys_ in keys_ids[1:]:
        print(len(batch_container[ii][keys_]))


# In[23]:


x_data = [ batch_container[ii]["data"] for ii in range(5)]
y_data = [ batch_container[ii]["labels"] for ii in range(5)]


# In[24]:


imgplot = plt.imshow(batch_container[0]["data"][120].reshape((32,32,3)),shape=(32,32,3))
batch_container[0]["labels"][120]


# In[25]:


from keras.datasets import cifar10

### Paramethers

num_classes = 10

####

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[26]:


imgplot = plt.imshow(x_train[120],shape=(32,32,3))

print(
x_train[120].shape,
y_train[120],)


# **Both image above are the same, but the first has being load from Disk and the other dowload directly from the Keras Repo. Dataset.**
# 
# > we are gonna work using the second loaded dataset

# In[27]:


#Architecture
def createModel(input_shape, nClasses):
    model = Sequential()
    
    model.add(Conv2D(36, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(36, (2, 2), activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
 
    model.add(Conv2D(48, (2, 2), padding='same', activation='selu'))
    model.add(Conv2D(48, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
 
    model.add(Flatten())
    model.add(Dense(500, activation='selu'))
    model.add(Dropout(0.8))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(nClasses, activation='softmax'))
     
    return model


# In[28]:


epochs = 150
batch_size = 1000

model1 = createModel(x_train.shape[1:], num_classes)

model1.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


# In[29]:


model1.summary()


# In[30]:


history = model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(x_test, y_test))

model1.evaluate(x_test, y_test)


# In[31]:


# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'.r')
plt.plot(history.history['val_loss'],'.b')
plt.plot(history.history['loss'],'--y')
plt.plot(history.history['val_loss'],'--g')
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'.r')
plt.plot(history.history['val_acc'],'.b')
plt.plot(history.history['acc'],'--y')
plt.plot(history.history['val_acc'],'--g')
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


# In[32]:


model1.save(filepath="./ann_model", include_optimizer=1)


# ### Continuing Training (Testing)

# In[59]:


model2 = copy(model1)


# In[61]:


batch_size = 5000
epochs = 5

history = model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(x_test, y_test))
model2.evaluate(x_test, y_test)


# In[62]:


# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'.r')
plt.plot(history.history['val_loss'],'.b')
plt.plot(history.history['loss'],'--y')
plt.plot(history.history['val_loss'],'--g')
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'.r')
plt.plot(history.history['val_acc'],'.b')
plt.plot(history.history['acc'],'--y')
plt.plot(history.history['val_acc'],'--g')
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


# In[63]:


model3 = copy(model2)


# In[70]:


batch_size = 2000
epochs = 50



history = model3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(x_test, y_test))
model3.evaluate(x_test, y_test)

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'.r')
plt.plot(history.history['val_loss'],'.b')
plt.plot(history.history['loss'],'--y')
plt.plot(history.history['val_loss'],'--g')
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'.r')
plt.plot(history.history['val_acc'],'.b')
plt.plot(history.history['acc'],'--y')
plt.plot(history.history['val_acc'],'--g')
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


# In[92]:


# Loss Curves
plt.figure(figsize=[8,6])

diff_loss = list(map(lambda x,y: (x-y), history.history['loss'] , history.history['val_loss']))
diff_loss_cum = np.cumsum(np.array(diff_loss)) / np.arange(1,51,1)

plt.plot(diff_loss,'.r')
plt.plot(diff_loss_cum, 'ob')

plt.plot(diff_loss,'--g')
plt.plot(diff_loss_cum, ':r')

plt.legend(['Diff', 'Mean Accumulatve Diff'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss (Train - Valid) Curves',fontsize=16)

# Acc Curves
plt.figure(figsize=[8,6])

diff_acc = list(map(lambda x,y: (x-y), history.history['acc'] , history.history['val_acc']))
diff_acc_cum = np.cumsum(np.array(diff_acc)) / np.arange(1,51,1)

plt.plot(diff_acc,'.r')
plt.plot(diff_acc_cum, 'ob')

plt.plot(diff_acc,'--g')
plt.plot(diff_acc_cum, ':r')

plt.legend(['Diff', 'Mean Accumulatve Diff'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Acc',fontsize=16)
plt.title('Acc (Train - Valid) Curves',fontsize=16)