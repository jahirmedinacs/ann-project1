
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt

import imageio

from copy import copy as copy
import time

import sys
import os
import random


# In[2]:


import keras


# In[3]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


# In[4]:


models = 5
model_container = [None for _ in range(models)]
for file_name in os.listdir():
    if "ann_model" in file_name:
        model_container[int(file_name[-1])] = file_name


# In[5]:


image_container = []
for file_name in os.listdir("./test"):
    image_container += [file_name]


# In[6]:


samples = 10
current_test_images = random.sample(image_container, samples)


# In[7]:


def menu(models, images):
    
    models_text = ""
    for ii in range(len(models)):
        models_text += "({:3d}) {:s} {:2s} {:s}\n".format(ii, models[ii], "\t", "CNN Pre-Trained Model")
    
    images_text = ""
    for ii in range(len(images)):
        img_class = images[ii][:-4].split("_")[1]
        images_text += "({:3d}) {:s} {:2s} {:s}\n".format(ii, img_class, "\t\t|\t", images[ii])
    
    do = True
    while do:
        print(
"""
\033[1m
\t\t  >>Models<<
[{:3d}] \t[Option] \t{:2s}[Description]
_________________________________________________\033[0m
{:s}

||=============================================||
\033[1m
\t\t  >>Images<<
[{:3d}] [File Class] {:2s}[Full Name File]
_________________________________________________\033[0m
{:s}

""".format(
    -1, "\t", models_text,
    -1, "\t", images_text))
        do = False


# In[8]:


repeat = True
while repeat:
    os.system('cls' if os.name == 'nt' else 'clear')
    menu(model_container, current_test_images)

    print(" Insert -1 to Exit ")

    try:
        model_id = int(input("\033[1m>>>Model?\t:\033[0m"))
        if model_id == -1:
            repeat = False
            break
        else:
            pass
        
        image_id = int(input("\033[1m>>>Image?\t:\033[0m"))
        if image_id == -1:
            repeat = False
            break
        else:
            pass
    except:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\t\033[1mInput Not Admited\033[0m\n")
        time.sleep(1)
    else:
        try:
            current_model = keras.models.load_model(filepath=model_container[model_id])
            current_image = imageio.imread("./test/" + current_test_images[image_id])
        except:
            break
        else:
            pass

            to_predict = np.ndarray((1,32,32,3))
            to_predict[0,] = current_image

            prediction = current_model.predict(to_predict)

            nice_prediction = np.zeros(10, dtype=int)
            nice_prediction[prediction.argmax()] = 1

            nice_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

            out_str = "\n\n\t " + str(nice_prediction.tolist()) + " is a {:s}\n\n".format(nice_names[prediction.argmax()])
            print("\033[1m" + out_str + "\033[0m")

            time.sleep(3)

print("\033[1m E X E C U T I O N   E N D E D \033[0m")

