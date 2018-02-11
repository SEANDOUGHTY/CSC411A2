from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Display the 150-th "5" digit from the training set
# imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
# show()

def download_num_imgs(M):

    # DOWNLOAD 10 IMAGES OF EACH NUMBER (0-9) FROM THE DATASET

    for key in M: # iterate through all the keys in the dict M with number images
        if key in ["__version__","__header__","__globals__"]: # we are only interested in keys containing image data
            continue
        
        num_imgs = []
        for i in range(0,10): # save and plot 10 example images for each number
            num_matrix = M[key][i].reshape((28,28)) # reshape the (174,) vector to a (28,28) matrix
            filename = key + "_" + str(i) + ".jpg" # want to save the image as a jpg file
            num_matrix = num_matrix/255.0
            mpimg.imsave("part1_photos/"+filename, num_matrix, cmap=cm.gray) # save the image
            num_imgs.append(num_matrix)
            num_in_img = key[-1] # get the number that the image is of
        display_imgs(num_imgs, num_in_img)

def display_imgs(num_imgs, num_in_img):

    # PLOT 10 IMGS OF THE SAME NUMBER

    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5)
    axes = ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10))
    i = 0
    j = 0
    for img in num_imgs:
        axes[i][j].imshow(img, cmap=cm.gray, aspect="equal")
        if j == len(axes[i])-1: # once first row of images is plot, reset the indices
            i,j = 1,-1
        j+=1
    ax3.set_title('10 Example Images of Number {} in the Dataset\n'.format(num_in_img))
    plt.show()


def network_compute(x,w,b):
    o = np.add(np.dot(w,x), b)
    
    return o


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    

#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################

def part1():

    print("____________________________________________________________")
    print "PART1: DOWNLOADING JPGS FROM mnist_all.mat FILE (NO OUTPUT)"
    print("____________________________________________________________")
    download_num_imgs(M)




############### RUNNING EACH PART ###############
#part1()















