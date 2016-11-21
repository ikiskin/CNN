import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# Keras-related imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling2D, Convolution2D
from keras import backend as K
K.set_image_dim_ordering('th')


##################################################################
dataset = np.load('CNNdata.npz')

x_train = dataset["arr_0"]
x_test_norm = dataset["arr_1"]  # Normalised version of x_test
x_test = dataset["arr_2"]
y_train = dataset["arr_3"]
y_test = dataset["arr_4"]
input_shape = dataset["arr_5"]
window_count = dataset["arr_6"]


# CNN parameters

# number of convolutional filters
nb_filters = 3

# size of pooling area for max pooling
pool_size = (2,2)

# convolution kernel size 
kernel_size = (5,5)

# number of classes
nb_classes = np.shape(y_train)[1]

# Initialise model

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                       border_mode = 'valid',
                       input_shape = input_shape))
model.add(Activation('relu'))

# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))

#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size = pool_size))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Train model

batch_size = 16
nb_epoch = 5

# For practice code run
# model.fit(X_train_set, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1, validation_split = 0.1) ## validation_data=(X_test_set, Y_test)
# score = model.evaluate(X_test_set, Y_test, verbose=0)
# predictions = model.predict(X_test_set)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# For spectrogram data run
x_train_log = 10*np.log10(x_train)
#x_test_log = np.log(x_test)
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_split = 0.1) ## validation_data=(X_test_set, Y_test)



score = model.evaluate(x_test_norm, y_test, verbose=1)
predictions = model.predict(x_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])



positive_predictions = predictions[:,0][np.where(y_test[:,0])]
negative_predictions = predictions[:,1][np.where(y_test[:,1])]

true_positive_rate = np.floor(sum(positive_predictions))/sum(y_test[:,0])
true_negative_rate = np.floor(sum(negative_predictions))/sum(y_test[:,1])

negative_sample_start = np.where(y_test[:,1])[0][0]


plt.figure(figsize = (15,6))
#plt.plot(y_test[:,0], 'b', linestyle = '--', linewidth = 2, label = 'y_test_positive')
plt.plot(predictions[:,0],'g.', label = 'y_pred_positive')
plt.plot(predictions[:,1], 'y.', label = 'y_pred_negative')
#[plt.axvline(_x, linewidth=1, color='kx') for _x in ([1,2,3])]

xp = 0
xn = negative_sample_start
for i in window_count:
    if xp < negative_sample_start:
        xp += i[0]
    else:
        xn += i[0]
    plt.axvline(x=xp, color = 'k', ls = 'dashed', linewidth = 0.5)
    plt.axvline(x=xn, color = 'r', ls = '--', linewidth = 0.5)
plt.axvline(x=negative_sample_start, color = 'b', ls = '-', linewidth = 2, label = 'Class boundary')

    
plt.legend(loc = 7)
plt.ylim([-0.2,1.2])
plt.ylabel('Softmax output')
plt.xlabel('Signal window number')
plt.show()
plt.savefig('ClassifierOut.png')
print 'True positive rate', true_positive_rate, 'True negative rate', true_negative_rate
model_name = 'my_model.h5'
model.save(model_name) # Save the model (see documentation model.save())
print 'Model saved as', model_name
