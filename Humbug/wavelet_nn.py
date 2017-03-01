#import audacity
import numpy as np
import sys 
import os, os.path
from scipy.io.wavfile import read, write
from scipy import signal
from scipy import nanmean
import csv
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from collections import Counter

# Keras-related imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling2D, Convolution2D
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint
from keras.callbacks import RemoteMonitor
from keras.models import load_model

# Data post-processing and analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

################################################################################################################################

## Load data

loaded_data = np.load('HumBug/Outputs/humbug_MLP_wavelet_80_10_10_most_positive_interp.npz')
print 'Files available to load:', loaded_data.files
x_train = loaded_data['x_train']
y_train = loaded_data['y_train']
x_test = loaded_data['x_test']
y_test = loaded_data['y_test']

x_train_caged = x_train_wav
y_train_caged = y_train_wav


# NN parameters
conv = False

# number of convolutional filters
nb_filters = 16

# size of pooling area for max pooling
pool_size = (2,2)

# convolution kernel size 
kernel_size_1 = (spec_window,spec_window)
kernel_size_2 = (3,3)
# number of classes
nb_classes = 2

# Initialise model

model = Sequential()

# Fully connected first layer to replace conv layer
n_hidden = 32 # N.B. Not sure exactly if this is the number of units in the hidden layer

model.add(Dense(n_hidden, input_dim=np.shape(x_train_caged)[1]))


# model.add(Convolution2D(nb_filters, kernel_size_1[0], kernel_size_1[1],
#                        border_mode = 'valid',
#                        input_shape = input_shape))
# convout1 = Activation('relu')
# model.add(convout1)
# model.add(Convolution2D(nb_filters, kernel_size_2[0], kernel_size_2[1]))

# convout2 = Activation('relu')
# model.add(convout2)
# model.add(MaxPooling2D(pool_size = pool_size))
# model.add(Dropout(0.25))

#model.add(Flatten())
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


################################################################################################################################
# Train model

#y_test = y_test_spec
#x_test = x_test_spec
#x_train_caged = x_train
#y_train_caged = y_train_wav
#input_shape = (1, x_train.shape[2], 10)
# # Reshape data for MLP
if not conv:
    x_train_caged = x_train.reshape(x_train.shape[0], x_train.shape[-2]*x_train.shape[-1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[-2]*x_test.shape[-1])
    y_train_caged = y_train_wav

    

batch_size = 64
nb_epoch = 200

# filepath = "weights-improvement.hdf5"
# #filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

# # Options to save best performing iteration according to monitor = ?. E.g. 'val_acc' will save the run with the highest 
# # validation accuracy.

# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') 
# callbacks_list = [checkpoint]
# remote = RemoteMonitor(root='http://localhost:9000') # For viewing accuracy measures during training. Experimental.

print np.shape(x_train_caged)
Weights = np.zeros([nb_epoch,np.shape(x_train_caged)[1],n_hidden])
for i in range(nb_epoch):
    print 'Epoch number', i+1, 'of', nb_epoch
    model.fit(x_train_caged, y_train_caged, batch_size=batch_size, nb_epoch=1, 
              verbose=2)
    W = model.layers[0].W.get_value(borrow=True)
    Weights[i,:,:] = W
    
#model.fit(x_train_caged, y_train_caged, batch_size=batch_size, nb_epoch=nb_epoch,
#         verbose=2)#,callbacks= [remote]) ## validation_data=(X_test_set, Y_test)
#RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers={'Content-Type': 'application/json', 'Accept': 'application/json'})

# Set file name for wavelet
base_name = 'humbug'
suffix_2 = 'wavelet'
if conv:
    suffix = 'conv'
else:
    suffix = 'MLP'
model_name = (base_name + '_' + suffix + '_' + suffix_2 + '_' + str(len(scales)) + '_' 
              + str(kernel_size_1[0]) + '_' + str(kernel_size_1[0]) + '_' + count_method + '_' + binning_method)
print model_name



score = model.evaluate(x_test, y_test, verbose=1)
predictions = model.predict(x_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

########### 2 class predictions #####################################################
positive_predictions = predictions[:,0][np.where(y_test[:,0])]
negative_predictions = predictions[:,1][np.where(y_test[:,1])]

true_positive_rate = (sum(np.round(positive_predictions)))/sum(y_test[:,0])
true_negative_rate = sum(np.round(negative_predictions))/sum(y_test[:,1])



figs = []

f = plt.figure(figsize = (15,6))
plt.plot(predictions[:,0],'g.', markersize = 2, label = 'y_pred_positive')
plt.plot(y_test[:,0], '--b', linewidth = 0.5, markersize = 2, label = 'y_test_positive')
    
plt.legend(loc = 7)
plt.ylim([-0.2,1.2])
plt.ylabel('Softmax output')
plt.xlabel('Signal window number')

figs.append(f)
print 'True positive rate', true_positive_rate, 'True negative rate', true_negative_rate

#plt.savefig('Outputs/' + 'ClassOutput_' + model_name + '.pdf', transparent = True)
#print 'saved as', 'ClassOutput_' + model_name + '.pdf' 
#plt.show()


cnf_matrix = confusion_matrix(y_test[:,1], np.round(predictions[:,1]).astype(int))
class_names = ('Mozz','No mozz')
# Plot normalized confusion matrix

f, axs = plt.subplots(1,2,figsize=(12,6))
#plt.figure(figsize = (4,4))
#plt.subplot(1,2,1)
#plt.figure(figsize = (4,4))
conf_m = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',)

#plt.savefig('Outputs/' + 'Conf_' + model_name + '.pdf', transparent = True)
#print 'saved as', 'Conf_' + model_name + '.pdf' 

y_true = y_test[:,0]
y_score = predictions[:,0]
roc_score = roc_auc_score(y_true, y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)


#plt.subplot(1,2,2)
#plt.figure(figsize=(4,4))
axs[0].plot(fpr, tpr, '.-')
axs[0].plot([0,1],[0,1],'k--')
axs[0].set_xlim([-0.01, 1.01])
axs[0].set_ylim([-0.01, 1.01])
axs[0].set_xlabel('False positive rate')
axs[0].set_ylabel('True positive rate')
axs[0].set_title('ROC, area = %.4f'%roc_score)
#plt.savefig('Outputs/' + 'ROC_' + model_name + '.pdf')
#print 'saved as', 'ROC_' + model_name + '.pdf' 
#plt.show()
figs.append(f)


pdf = matplotlib.backends.backend_pdf.PdfPages('Outputs/' + model_name + '.pdf')
for i in range(2): 
    pdf.savefig(figs[i])
pdf.close()
print 'saved as ' + model_name + '.pdf'