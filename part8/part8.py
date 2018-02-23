from pylab import *
import numpy as np
from torch.autograd import Variable
import torch
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import re # added
import os # added
import sys # added


act = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))    
testfile = urllib.URLopener()
image_files = []
photo_num = dict([("Bracco",0), ("Gilpin",0), ("Harmon",0), ("Baldwin",0), ("Hader",0), ("Carell",0), ("Radcliffe",0), ("Butler",0), ("Vartan",0), ("Chenoweth",0), ("Drescher",0), ("Ferrera",0)])                                                                       

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def download_jpgs_from_txt(file):
    # --FUNCTION--: 
    # accesses the photo information from file source and downloads the image
    # into a folder called "uncropped"
    #--INPUT--
    # file, a text file containing the dataset of actor/actress image information

    for line in open(file).readlines():
        data = line.split("\t")
        if desired_actor(data) == 0:    # breaks the loop if it is an actor we are not looking for
            continue

        for info in data:
            if info.endswith(".jpg") or info.endswith(".JPG") or info.endswith(".png"):   # checks each item in the list and puts all the jpgs in a list
                url = info
                image_files.append(info) # might not need this anymore

        actor_name = data[0].split(" ")  # splits the actors name into a 2 item list ["first", "last"]
        
        filename = actor_name[1] + str(photo_num[actor_name[1]]) + ".jpg"
        dimensions = [int(data[1]), int(data[2])] # converts string dimensions of photo in data to integers

        timeout(testfile.retrieve, (url, "uncropped/" + filename), {}, 30)
        
        try:
            crop_dimensions = []             # creates an empty list to which we append after splitting the data string of dims into integers            
            for dim in data[4].split(','):
                crop_dimensions.append(int(dim))

            img_matrix= imread("uncropped/"+filename)
            crop_gray_resize_jpgs(filename, crop_dimensions, actor_name) # actor_name is given because we add to global photo_num dictionary inside the rop_gray_resize_jpgs() function

        except Exception as e:
            print(e)
            crop_dimensions = "INVALID"

        mod_data = [url, filename, dimensions, crop_dimensions] # might not need this

def desired_actor(data):
    # --FUNCTION--: 
    # returns 1 if data[0] is a desired actor, returns 0 otherwise
    #--INPUT--
    # data, an array of information given from each line in the dataset
    if data[0] in ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell"]:
        return 1
    return 0

def crop_gray_resize_jpgs(filename, crop_dimensions, actor_name):
    # --FUNCTION--: 
    # Crops, grays, and resizes filename (an image) to crop_dimensions and then
    # saves it in a folder titled "cropped"
    #--INPUT--
    # filename is a string filename of the image being cropped
    # crop_dimensions is a tuple of the (x,y) coordinates to crop the image at
    # actor_name is a string of the actor's last name


    img_matrix = imread("uncropped/"+filename)

    x1 = crop_dimensions[0]
    y1 = crop_dimensions[1]
    x2 = crop_dimensions[2]
    y2 = crop_dimensions[3]

    c_matrix = img_matrix[y1:y2,x1:x2]     # cropped matrix

    # some images are already gray-scaled so the rgb2gray function does not work
    # this if else statement checks if it is rgb and if so, makes it gray-scaled

    # DON'T GRAYSCALE HERE
    # if size(c_matrix.shape) == 3:
    #     cg_matrix = rgb2gray(c_matrix)     # cropped grayscale matrix
    # else:
    #     cg_matrix = c_matrix
    cg_matrix = c_matrix
    cgr_matrix = imresize(cg_matrix, (32,32)) # cropped grayscale resized matrix
    
    photo_num[actor_name[1]] += 1 # add one to the actor's key every time they get a photo saved
    # print("saving:" + filename)
    # print("WHILE SAVING:")
    # print("size:" +str(cgr_matrix.size))
    # print("shape:" + str(cgr_matrix.shape))
    # print("image matrix of"+str(filename) + "is" + str(cgr_matrix))

    mpimg.imsave("cropped_rgb/"+filename, cgr_matrix, cmap=plt.cm.gray) # save the image
    
    #TO SHOW AN IMAGE (IN GRAY SCALE) FROM ITS IMAGE MATRIX USE:
    #matplotlib.pyplot.imshow(cg_matrix,cmap=plt.cm.gray)
    #plt.show()

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def create_sets(path, actors, train_set_len = 55, val_set_len = 10, test_set_len = 20):
    # --FUNCTION--: 
    # Returns three sets of data - training, validation, and test - containing
    # 70, 10, and 10 photos for each actor, respectively. 
    #--INPUT--
    # path is a string of the path to the folder of cropped photos

    # declare temp sets that will be added throughout the function
    i=1
    training_set = set([])
    validation_set = set([])
    test_set = set([])

    temp_training_set = set([])
    temp_validation_set = set([])
    temp_test_set = set([])
    
    folder = os.listdir(path) # folder = "cropped/"

    # initiate a full list of the actors
    # each actor will be removed one at a time as their photos are added to each set
    actors = ["Bracco", "Gilpin", "Harmon", "Baldwin", "Hader", "Carell"]
    #print("folder[i]:" + str(folder[i]))
    #print("shape:" + str(imread("cropped/"+folder[i], True).shape))
    #print("SETS: image matrix of"+str(folder[i]) + "is" + str(imread("cropped/"+folder[i])))
    while actors!= []:
        # first line in this loop splits the file name into a list with str and ints divided
        # i.e. img = "Baldwin0.jpg" would return ['Baldwin','0','.jpg']             
        name = re.split('(\d+)',folder[i])[0]  # gets the name of the actor associated with the img

        # while the actor name is still the same, fill each set
        # once all sets are full, we iterate with different actor
        # return once index reaches length of folder
        while re.split('(\d+)',folder[i])[0] == name:
            if len(temp_training_set) != train_set_len:
                temp_training_set.add(folder[i])
            elif len(temp_validation_set) != val_set_len:
                temp_validation_set.add(folder[i])
            elif len(temp_test_set) != test_set_len:
                temp_test_set.add(folder[i])
            i += 1
            if i == len(folder):
                training_set = training_set.union(temp_training_set)
                validation_set = validation_set.union(temp_validation_set)
                test_set = test_set.union(temp_test_set)
                return training_set, validation_set, test_set
       
        #ADD EACH ACTOR'S PHOTOS TO THE SET
        training_set = training_set.union(temp_training_set)
        validation_set = validation_set.union(temp_validation_set)
        test_set = test_set.union(temp_test_set)

        #RESET THE TEMP SETS
        temp_training_set = set([])
        temp_validation_set = set([])
        temp_test_set = set([])
        actors.remove(name)

def get_x_and_y_data(dataset):
        # ONE-HOT ENCODING
        # "Hader" =  [1 0 0 0 0 0]
        # "Gilpin" = [0 1 0 0 0 0]
        # "Harmon" = [0 0 1 0 0 0]
        # "Baldwin" =[0 0 0 1 0 0]
        # "Bracco" = [0 0 0 0 1 0]
        # "Carell" = [0 0 0 0 0 1]
        index_and_label = {}
        num = 0
        x = np.ndarray([1])
        y = np.ndarray([1])

        for img in dataset:  # A label is added to the y matrix corresponding to its respective image in the x matrix
            if "Hader" in img:
                label = [1, 0, 0, 0, 0, 0]
                add = True

            if "Gilpin" in img:
                label = [0, 1, 0, 0, 0, 0]
                add = True

            if "Harmon" in img:
                label = [0, 0, 1, 0, 0, 0]
                add = True

            if "Baldwin" in img:
                label = [0, 0, 0, 1, 0, 0]
                add = True

            if "Bracco" in img:
                label = [0, 0, 0, 0, 1, 0]
                add = True

            if "Carell" in img:
                label = [0, 0, 0, 0, 0, 1]
                add = True

            if add:   
                img_matrix = imread("cropped_rgb/"+img, True)
                img_matrix = img_matrix/255 # normalize the matrix values to lie between 0 and 1

                data = np.ndarray([]) # create an empty data array

                i = 0
                while i < 32: # cycle until all 32 rows of the matrix have been added to data
                    row = img_matrix[:i+1,:][i]
                    if data.size == 1:
                        data = row
                    else:
                        data = np.concatenate([data,row]) # add each row of pixels to the data vector
                    i+=1

                if x.size == 1:
                    x = np.array(data)
                else:
                    x = np.vstack((x,data))

                if y.size == 1:
                    y = np.array(label)
                else:
                    y = np.vstack((y,label))

                index_and_label[num] = [label]
                num+=1


            add = False

        # take the transpose because we want each label and data vector to be vertical
        # in the matrices
        y = y.T # dimension: 6 x 330
        x = x.T # dimension: 1024 x 330


        # we must account for the theta0 term so we add an unity element to x_matrix
        # x_matrix.shape[1] ensures we add the right size vector (140 for training set and 20 for validation/test)
        unity_row = np.ones((1,x.shape[1]))
        x = np.vstack((x, unity_row)) # dimension: 1025 x 420

        return x.T, y.T

def part1():
    print("_________________________________________________________")
    print("PART1: DOWNLOADING JPGS FROM TEXT (NO OUTPUT)")
    print("_________________________________________________________")
    download_jpgs_from_txt("subset_actors.txt")

def part2():
    print("_________________________________________________________")
    print("PART2: PARTIONIONING ACTORS INFO INTO SETS (NO OUTPUT)")
    print("_________________________________________________________")
    actors = ["Bracco", "Gilpin", "Harmon", "Baldwin", "Hader", "Carell"]
    training_set, validation_set, test_set = create_sets("cropped_rgb/", actors)

    train_x, train_y = get_x_and_y_data(training_set)
    test_x, test_y = get_x_and_y_data(test_set)

    # print("Shape of train_x, train_y: ", train_x.shape, train_y.shape)
    # print("Shape of test_x, test_y: ", test_x.shape, test_y.shape)

    dim_x = 32*32 + 1 # CHANGED THIS
    dim_h = 20
    dim_out = 6 # CHANGED THIS

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

	################################################################################
	#Subsample the training set for faster training
    train_idx = np.random.permutation(range(train_x.shape[0]))[:55] # CHANGED THIS FROM 1000
    x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)
	#################################################################################

    # DEFINE THE NEURAL NETWORK
    model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),torch.nn.ReLU(),torch.nn.Linear(dim_h, dim_out),)

    # INITIALIZE WEIGHTS
    def init_weights(m):
    	# print(m)
    	if type(m) == torch.nn.Linear:
    		m.weight.data.fill_(0.01)

    def test_model(x_set, y_set):
    	# MAKE PREDICTIONS FOR THE SET DATA
    	x = Variable(torch.from_numpy(x_set), requires_grad=False).type(dtype_float)
    	y_pred = model(x).data.numpy()

    	# LOOK AT THE PERFORMANCE
    	accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(train_y, 1))
    	return accuracy

    model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),torch.nn.ReLU(),torch.nn.Linear(dim_h, dim_out),)
    model.apply(init_weights)

    # DEFINE A LOSS FUNCTION
    loss_fn = torch.nn.CrossEntropyLoss()

    # TRAIN THE MODEL USING ADAM, A VARIANT OF GRADIENT DESCENT
    learning_rate = 0.00001 #1e-2

    # CHOOSE AN OPTIMIZER
    # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iterations = 100000

    loss_data = [[],[]]
    for t in range(iterations): # MIGHT NEED TO CHANGE THIS FROM 1000 to 120 (10000 originally)
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                           # make a step                        
        if t%500 == 0:
        	print("Optimization is {}% complete.".format(100*t/iterations))
        	print("Accuracy: {}".format(test_model(train_x,train_y)))
        	started = True
       	if t%100 == 0:
	       	loss_data[0].append(t)
	       	loss_data[1].append(float(loss))

    if started: print("Optimization is 100% complete.")

    def plot_loss_vs_iterations(loss_data):
        # plt.yscale('log')
        plt.title("Loss vs. Iterations")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(loss_data[0], loss_data[1], '-b', label="Loss")
        plt.legend(loc='best')
        plt.show()

    # PLOT THE LOSS OVER ITERATIONS
    # plot_loss_vs_iterations(loss_data)

    # # YOU CAN ACCESS THE WEIGHTS LIKE THIS
    # model[0].weight

    # # HAVE TO GET RID OF THE BIAS ELEMENT (i.e., convert from dim 1025 to dim 1024)
    # model_array = model[0].weight.data.numpy()
    # model_array = model_array[:,:-1]

    # # PLOT THE WEIGHTS ASSOCIATED WITH UNIT 10
    # model_unit_weights = model_array[10, :]
    # plt.imshow(model_unit_weights.reshape((32, 32)), cmap=plt.cm.coolwarm) # CHANGED THIS FROM 28,28
    # # plt.show()


    # # PLOT THE WEIGHTS ASSOCIATED WITH UNIT 12
    # model_unit_weights = model_array[18, :]
    # plt.imshow(model_unit_weights.reshape((32, 32)), cmap=plt.cm.coolwarm) # CHANGED THIS FROM 28,28
    # # plt.show()

############### RUNNING EACH PART ###############
# part1()
part2()
