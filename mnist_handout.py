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
import copy

import cPickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Display the 150-th "5" digit from the training set
# imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
# show()

def make_set(M, typ, size=-1):
    load = load_set(typ)
    if len(load) == 2: # check if the set is already downloaded and saved
        print('Loading ' + typ + ' Set from Memory.')
        return load

    new_setx = np.zeros(shape=(1,785)) # 784 pixels + a 1 for the bias
    new_sety = np.zeros(shape=(1,10)) # 10 possible out puts
    arraysize = copy.deepcopy(size)

    for key in M:
        if key in ["__version__","__header__","__globals__"]: # we are only interested in keys containing image data
            continue
        if typ not in key:  #only interest in train or test
            continue
        if size == -1: #if no size parameter then use the full training set
            arraysize = len(M[key])-1        

        print('Making ' + typ + ' set: ' + str(key) + '.')    
        number = int(key[-1])
        for i in range(arraysize):
            num_matrix = M[key][i].reshape((28,28)) # reshape the (174,) vector to a (28,28) matrix
            num_matrix = num_matrix/255.0
            if i % 1000 == 0:
                print(str(float(i)/float(arraysize)*100) + '%')

            k = 0
            for i in num_matrix: #converting to a vector
                for j in i:
                    new_setx[-1][k] = j
                    k += 1
            new_setx[-1][-1] = 1
            
            new_sety[-1][number] = 1

            new_setx = np.vstack([new_setx, np.zeros(785)]) #initalizting the next row
            new_sety = np.vstack([new_sety, np.zeros(10)]) #initalizting the next row

    new_setx = new_setx[:-1] #removing the empty final row
    new_sety = new_sety[:-1]
    np.save((typ+'_setx'), new_setx)
    np.save((typ+'_sety'), new_sety)
    print(new_setx.shape)
    print(new_sety.shape)
    return [new_setx,new_sety]

def load_set(typ):
    train_set = []
    if ((os.path.isfile(typ + '_setx.npy')) and (os.path.isfile(typ + '_sety.npy'))):
        train_set.append(np.load(typ + '_setx.npy'))
        train_set.append(np.load(typ + '_sety.npy'))
    return train_set

def initalize_weights():
    if ((os.path.isfile('weights.npy'))):
        print('Loading Trained Weights from Memory')
        return np.load('weights.npy')
    print('Initalizting Weights')
    w = np.ones(shape = (10,785)) #final element is the bias, intializing the weights to 0.5
    w = 0.5*w
    return w

def df(x,y,p):
    #formula: x(p-y)
    return np.dot(x.transpose(), np.subtract(p.transpose(),y))

def grad_descent(df, x, y, x1, y1, init_t, alpha, gamma, iterations):
    # x, y are the training sets
    # x1, y1 are the test sets
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = iterations
    iter  = 0
    last_grad = 0

    while norm(t - prev_t) >  EPS and iter < max_iter:

        prev_t = t.copy()

        p = network_compute(x,t)

        grad = (alpha*df(x, y, p)).transpose()
        t -= gamma*last_grad + grad
        last_grad = grad
        if iter % 500 == 0:
            print "Iter:", iter
            trainresults = check_results(x,y,t)
            testresults = check_results(x1,y1,t)
            print ("Training: " + str(trainresults) + "%")
            print ("Testing: " + str(testresults) + "%")

            name = raw_input("Continue?")
            if name == "N":
                break
        iter += 1
    np.save('weights.npy', t)
    return t

def check_results(x,y,t):
    '''This is a function that takes in a set of images, their result and the weights
    and calculates the accuracy of the network
    '''
    p = network_compute(x,t)
    correct = 0
    incorrect = 0
    
    for i in range(y.shape[0]):
        if np.argmax(y[i][:]) == np.argmax(p.transpose()[i]): #checking which images get choses correctly
            correct += 1
            continue
        incorrect += 1

    if (incorrect + correct) == 0:
        return 0
    return float(correct)/float((incorrect+correct))*100

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

def network_compute(x,w):
    o = np.dot(w,x.transpose())
    p = softmax(o)
    return p

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

def part4():
    print("____________________________________________________________")
    print "PART4: TRAINING NETWORK (WITHOUT MOMENTUM)"
    print("____________________________________________________________")

    train_set = make_set(M, 'train') #building the training and test sets
    test_set = make_set(M, 'test')
    w = initalize_weights()

        
    alpha = 0.001
    iterations = 10000        
    momentum = 0
    t = grad_descent(df, train_set[0], train_set[1], test_set[0], test_set[1], w, alpha, momentum, iterations)
    t = load('weights.npy')
    result = check_results(train_set[0],train_set[1],t)
    print(str(result)+'%')
    result = check_results(test_set[0],test_set[1],t)
    print(str(result)+'%')

def part5():
    print("____________________________________________________________")
    print "PART5: TRAINING NETWORK (WITH MOMENTUM)"
    print("____________________________________________________________")
    train_set = make_set(M, 'train') #building the training and test sets
    test_set = make_set(M, 'test')
    w = initalize_weights()
        
    alpha = 0.00001
    iterations = 100000        
    momentum = 0.99
    t = grad_descent(df, train_set[0], train_set[1], test_set[0], test_set[1], w, alpha, momentum, iterations)
    t = load('weights.npy')
    result = check_results(train_set[0],train_set[1],t)
    print(str(result)+'%')
    result = check_results(test_set[0],test_set[1],t)
    print(str(result)+'%')

############### RUNNING EACH PART ###############
#part1()
#part4()
part5()















