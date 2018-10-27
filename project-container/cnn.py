import matplotlib.pyplot as plt

import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.datasets import cifar10

### Pseudo Macros

PLOT = False
VERBOSE = 0

NUM_OF_CLASS = 10 # CIFAR - 10

####

# The data, split between train and test sets:
(Input_train, Objetive_train), (Input_test, Objetive_test) = cifar10.load_data()
print('Input_train shape:', Input_train.shape)
print(Input_train.shape[0], 'train samples')
print(Input_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Objetive_train = keras.utils.to_categorical(Objetive_train, NUM_OF_CLASS)
Objetive_test = keras.utils.to_categorical(Objetive_test, NUM_OF_CLASS)


# Testing Data Load Correctnes
imgplot = plt.imshow(Input_train[120],shape=(32,32,3))

try:
    print("Load Succefully : Sample >>\t",
        Input_train[120].shape,
        Objetive_train[120],
        "\n\n")
except:
    print("Error Loading Data, Exiting Program...")
    return 0
    end()
else:
    pass

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

## Train Parameters

epochs = 150
batch_size = 1000

###

output_model = createModel(Input_train.shape[1:], NUM_OF_CLASS)
output_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

#output_model.summary()
if PLOT:
    history = output_model.fit(Input_train, Objetive_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE, 
                       validation_data=(Input_test, Objetive_test))
else:
    output_model.fit(Input_train, Objetive_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE, 
                   validation_data=(Input_test, Objetive_test))

output_model.evaluate(Input_test, Objetive_test)
output_model.save(filepath="./cnn_model_v0", include_optimizer=1)

if PLOT:
    ## Plots
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
    plt.show()

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
    plt.show()
else:
    pass