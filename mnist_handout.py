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

def f(x,y,w):
    p = network_compute(x,w)
    logp = np.log(p)


    cost = np.dot(y,logp)

    cost = -np.trace(cost)
    #print(cost)
    return cost


def df(x,y,p):
    #formula: x(p-y)
    return np.dot(x.transpose(), np.subtract(p.transpose(),y))


def grad_descent(df, x, y, x1, y1, init_t, alpha, gamma, iterations, frequency=500, interupts=True):
    EPS = 1e-5   #EPS = 10**(-5)
    itertrack = np.array([])
    traintrack = np.array([])
    testtrack = np.array([])

    itercount = np.array([])
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

        if iter % frequency == 0:
            print "Iter", iter
            trainresults = check_results(x,y,t)
            testresults = check_results(x1,y1,t)
            itertrack = np.append(itertrack, iter)
            traintrack = np.append(traintrack, trainresults)
            testtrack = np.append(testtrack, testresults)
            print ("Training: " + str(trainresults) + "%")
            print ("Testing: " + str(testresults) + "%")


            if interupts:
                name = raw_input("Continue")
                if name == "N":
                    break
        iter += 1
    np.save('weights.npy', t)
    return [itertrack, traintrack, testtrack]


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

        
    alpha = 0.00001
    iterations = 100000       
    momentum = 0
    frequency = 500
 
    results = grad_descent(df, train_set[0], train_set[1], test_set[0], test_set[1], w, alpha, \
        momentum, iterations, frequency, True)  
    
    
    plt.plot(results[0], results[2], 'b', results[0], results[1], 'g')
    plt.xlabel('Iterations')
    plt.ylabel('Test Set Accuracy %')
    plt.show()

    return results

def part4visualize():
    w = np.load('weights.npy')
    
    i = 0
    for output in w: # save and plot each visualization
        num_matrix = output[:-1].reshape((28,28)) # reshape the (174,) vector to a (28,28) matrix
        filename = "visualize" + str(i) + ".png" # want to save the image as a jpg file
        num_matrix = num_matrix/255.0
        mpimg.imsave("part4-5photos/"+filename, num_matrix, cmap=cm.gray) # save the image
        i += 1

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
    frequency = 1
 
    results = grad_descent(df, train_set[0], train_set[1], test_set[0], test_set[1], w, alpha, \
        momentum, iterations, frequency, False)  
    

    plt.plot(results[0], results[2], 'b', results[0], results[1], 'g')
    plt.xlabel('Iterations')
    plt.ylabel('Test Set Accuracy %')
    plt.show()

    return results

def part6():
    w = np.load('weights.npy')
    x = np.load('test_setx.npy')
    y = np.load('test_sety.npy')


    #gd_traj = [(init_w1, init_w2), (step1_w1, step1_w2), ...]
    #mo_traj = [(init_w1, init_w2), (step1_w1, step1_w2), ...]
    w1s = np.arange(-0, 1, 0.1)
    w2s = np.arange(-0, 1, 0.1)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    D = np.zeros([w1s.size, w2s.size])
    k = 0
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            print(str(i) + ':' + str(w1) + ' ' + str(w2))
            #w[2][625] = w1
            w[2][596] = w2
            p = network_compute(x,w)
            print('grad' +str(df(x,y,p)[596][2]))
            C[j,i] = f(x,y,w)
            #C[i,j] = k
            k += 1

    #C = C - C.mean()
    #A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    #np.save('test.npy', C)
    #C = np.zeros([w1s.size, w2s.size])
    #C[1,1] =
    print(C)
    #CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
    #plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    #plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    #return 0
    plt.contour(C)
    plt.xlim( (0, 3) )
    plt.ylim( (0, 3) )
    #plt.legend(loc='top left')
    plt.title('Contour plot')
    plt.show()
    
        

    
    



############### RUNNING EACH PART ###############
#part1()
#os.remove("weights.npy")
#part4()
#part4visualize()
#part5()
part6()

'''
C = np.load('weights.npy')
i = 2
for j in range(len (C[2])):
    print('number' + str(j) + ':' + str(C[i][j]))
'''
















