#from __future__ import print_function
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io.wavfile import read, write
import os, os.path
from scipy import signal
import IPython
np.random.seed(1337)  # for reproducibility

import random



###############################################################################################################

def proc_data(path, img_height, img_width, nfft = 512, overlap = 128, normalise = False):
    """Supply a list of paths in path to return training data with number of classes equal to the length of the path list.
    Dataset named after string data_name."""
    data_dct = {}
    dirs = []
    
    for item in path:
        dirs.append(os.listdir(item))
    for index, item in enumerate(dirs):
        signal_list = []
        list_name = 'class_' + str(index)
        for file in dirs[index]:
            if file.endswith('.wav'):
                fs, signal = read(path[index] + file)
                if normalise:
                    print 'Normalising *.wav...'
                    signal_list.append(signal/np.std(signal))
                else:
                    signal_list.append(signal)
        data_dct.update({list_name: [signal_list]})
        
    
    ###################################################################################################################
    spec_list_positive = []
    spec_list_negative= []
    # Select whole list as training data (min: want equal number of recordings per class)
    n_train = min(np.shape(data_dct['class_0'])[1], np.shape(data_dct['class_1'])[1]) 
    #n_train = 5 # Use only n_train recordings as training data (2*n_train for positive and negative)
    for i in np.arange(n_train):
        sample = data_dct['class_0'][0][i]
        sample_negative = data_dct['class_1'][0][i]
        print 'Processing recording #', i
        SpecInfo = plt.specgram(sample, Fs = fs, NFFT = nfft, noverlap = overlap)
        spec_list_positive.append(SpecInfo)
        SpecInfo = plt.specgram(sample_negative, Fs = fs, NFFT = nfft, noverlap = overlap)
        spec_list_negative.append(SpecInfo)
    spec_list = {'class_0': spec_list_positive, 'class_1': spec_list_negative}
    print 'Number of recordings used as positive data', np.shape(spec_list_positive)
    print 'Number of recordings used as negative data', np.shape(spec_list_negative)
    print 'sample rate', fs
    
    ###################################################################################################################    
    # Dimensions for image fed into network
    nb_classes = 2
    x_train = []
    y_train_count = []
    window_count = []
    # Stack positive and negative training samples into a single array
    #spec_training_list = np.vstack([spec_training_list_positive, spec_training_list_negative]) 
    spec_training_list = spec_list['class_0'] + spec_list['class_1']
    # Format into same shape as mnist_cnn.py example
    for i in np.arange(len(spec_training_list)):
        n_max = np.floor(np.divide(np.shape(spec_training_list[i][0])[1],img_width)).astype('int')
        if i < len(spec_training_list)/2:
            window_count.append([n_max, 1])
        else:
            window_count.append([n_max, 0])
        print 'Processing signal number', i
        print 'Number of training inputs for this signal:', n_max    
        for n in np.arange(n_max):
            x_train.append(spec_training_list[i][0][:img_height,img_width*n:img_width*(n+1)])
            if i < len(spec_training_list)/2:
                y_train_count.append(np.array(1))
            else:
                y_train_count.append(np.array(0))

    x_train = np.array(x_train).reshape((np.shape(x_train)[0],1,img_height,img_width))

    y_train = np.zeros((len(y_train_count),nb_classes))

    y_train[np.where(y_train_count),0] = 1
    y_train[np.where(np.logical_not(y_train_count)),1] = 1
    y_positive_frac = np.shape(np.where(y_train_count))[1]/(1.*len(y_train))

    print '\nx dimensions', np.shape(x_train)
    print 'y dimensions', np.shape(y_train)
    print 'Fraction of positive samples', y_positive_frac

    input_shape = (1, img_height, img_width)
    
    return x_train, y_train, y_positive_frac, input_shape, window_count   

###############################################################################################################

def normalise(data, wrt_data):
    """Normalise Theano-shaped data arrays for 2D convolutional networks with respect to data in `wrt_data'"""
    var_data = np.var(wrt_data, axis = 0)

    return data/var_data, var_data


################################## Code Execution #############################################################

path_training = ['../../../data/BirdCLEF/training/wav/colibri/',
                 '../../../data/BirdCLEF/training/wav/other/']

x_train, y_train, y_positive_frac, input_shape, window_count_train =  proc_data(path_training, 256, 256, normalise = False)   

path_testing = ['../../../data/BirdCLEF/testing/wav/colibri/',
                 '../../../data/BirdCLEF/testing/wav/other/']
x_test, y_test, y_positive_frac_test, input_shape, window_count = proc_data(path_testing, 256, 256, normalise = False)

x_train_norm, x_train_var = normalise(x_train, x_train)
x_test_norm, x_train_test_var = normalise(x_test, x_train)

###############################################################################################################

np.savez('CNNdatanorm.npz', x_train_norm, x_test, x_test_norm, y_train, y_test, input_shape, window_count)
