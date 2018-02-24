from pylab import *
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
from scipy.io import loadmat
# import matplotlib.cbook as cbook
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
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

# RESIZE THE 32x32 CROPPED IMAGES TO 227x227
def resize_images(folder, dim):
    for file in os.listdir(folder):
        # fillename = str(file)
        if file != '.DS_Store':
            small_res_matrix = imread(folder + file)
            large_res_matrix = imresize(small_res_matrix, (dim,dim))
            mpimg.imsave("cropped_rgb_"+str(dim)+"/"+file, large_res_matrix) # save the image in another folder

# PARTITION THE DATA INTO DESIRED SIZES
def create_sets(path, actors, train_set_len = 80, val_set_len = 20, test_set_len = 20):
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

    while actors!= []:
        # first line in this loop splits the file name into a list with str and ints divided
        # i.e. img = "Baldwin0.jpg" would return ['Baldwin','0','.jpg']             
        name = re.split('(\d+)',folder[i])[0]  # gets the name of the actor associated with the img

        # while the actor name is still the same, fill each set
        # once all sets are full, we iterate with different actor
        # return once index reaches length of folder
        if name == "Gilpin":
            train_len = int(train_set_len*.75)
            val_len = int(val_set_len*.75)
            test_len = int(test_set_len*.75)
        else:
            train_len = train_set_len
            val_len = val_set_len
            test_len = test_set_len

        while re.split('(\d+)',folder[i])[0] == name:
            if len(temp_training_set) != train_len:
                temp_training_set.add(folder[i])
            elif len(temp_validation_set) != val_len:
                temp_validation_set.add(folder[i])
            elif len(temp_test_set) != test_len:
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

# OBTAIN TARGET Y VECTORS WITH THE APPROPRIATE ONE-HOT ENCODINGS ASSOCIATED WITH IMAGES IN X
def get_x_and_y_data(dataset,folder):
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
            img_matrix = imread(folder+img, True)
            img_matrix = img_matrix/255.0 # normalize the matrix values to lie between 0 and 1
            data = np.ndarray([]) # create an empty data array

            i = 0
            while i < 227: # cycle until all 227 rows of the matrix have been added to data
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
    y = y.T # dimension: 6 x N
    x = x.T # dimension: 51529 x N

    # we must account for the theta0 term so we add an unity element to x_matrix
    # x_matrix.shape[1] ensures we add the right size vector (140 for training set and 20 for validation/test)
    unity_row = np.ones((1,x.shape[1]))
    x = np.vstack((x, unity_row)) # dimension: 1025 x 420

    return x.T, y.T

# INITIALIZE WEIGHTS
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.000001)

# SUBSAMPLE THE TRAINING SET FOR FASTER TRAINING
def subsample(train_x,train_y,batch_size):
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    train_idx = np.random.permutation(range(train_x.shape[0]))[:batch_size] # CHANGED THIS FROM 1000
    x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)
    return x, y_classes

# TEST THE MODEL ON AN INPUT SET (EITHER TRAINING, VALIDATION, OR TEST)
def test_model(model, x_set, y_set):
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    # MAKE PREDICTIONS FOR THE SET DATA
    x = Variable(torch.from_numpy(x_set), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()

    # LOOK AT THE PERFORMANCE
    accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(y_set, 1))
    return accuracy

# BUILD, TRAIN, AND TEST THE NETWORK
def train_and_test(model,train_x,train_y,val_x,val_y,test_x,test_y,batch_size=45,lr=0.0005):

    # DEFINE A LOSS FUNCTION
    loss_fn = torch.nn.CrossEntropyLoss()

    # CHOOSE AN OPTIMIZER
    # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

    # TRAIN THE MODEL USING ADAM, A VARIANT OF GRADIENT DESCENT
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iterations = 800
    loss_data = [[],[]]
    accuracy_data = {"train_set": [[],[]], "val_set": [[],[]], "test_set": [[],[]]}

    for t in range(iterations): # MIGHT NEED TO CHANGE THIS FROM 1000 to 120 (10000 originally)
        x, y_classes = subsample(train_x,train_y,batch_size)
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to make a step   

        # PROVIDE UPDATES
        if t%100 == 0:
            print("Optimization is {}% complete.".format(100*t/iterations))
            print("Accuracy for Test Set: {}".format(test_model(model,test_x,test_y)))
            started = True

        # COLLECT DATA
        if t%50 == 0:
            loss_data[0].append(t)
            loss_data[1].append(float(loss))

            # COLLECT TRAINING DATA
            accuracy_data["train_set"][0].append(t)
            accuracy_data["train_set"][1].append(test_model(model,train_x,train_y))

            # COLLECT VALIDATION DATA
            accuracy_data["val_set"][0].append(t)
            accuracy_data["val_set"][1].append(test_model(model,val_x,val_y))

            # COLLECT TEST DATA
            accuracy_data["test_set"][0].append(t)
            accuracy_data["test_set"][1].append(test_model(model,test_x,test_y))

    if started: 
    	print("Optimization is 100% complete.")
    	print("Accuracy for Test Set: {}".format(test_model(model,test_x,test_y)))

    return accuracy_data, loss_data

# SAVE THE NUMPY ARRAYS AFTER FETCHING THE X AND Y DATA TO SAVE TIME
def save_numpy_arrays(train_x, train_y, test_x, test_y, val_x, val_y):
	np.save("train_x.npy", train_x)
	np.save("train_y.npy", train_y)
	np.save("test_x.npy", test_x)
	np.save("test_y.npy", test_y)
	np.save("val_x.npy", val_x)
	np.save("val_y.npy", val_y)

# SAVE THE TENSORS TO SAVE TIME EACH RUN
def save_tensors(train_x_tensor, test_x_tensor, val_x_tensor):
	torch.save(train_x_tensor, "train_x_tensor.pth")
	torch.save(test_x_tensor, "test_x_tensor.pth")
	torch.save(val_x_tensor, "val_x_tensor.pth")

# LOAD TENSORS AND RETURN THEM AS NUMPY ARRAYS
def load_tensors_as_numpys():
	# LOAD TENSORS FROM FILES
	train_x_tensor = torch.load("train_x_tensor.pth")
	test_x_tensor = torch.load("test_x_tensor.pth")
	val_x_tensor = torch.load("val_x_tensor.pth")

	# CONVERT LOADED TENSORS TO NUMPY ARRAYS
	train_x = train_x_tensor.data.numpy() # dim = (460, 43264)
	test_x = test_x_tensor.data.numpy() # dim = (115, 43264)
	val_x = val_x_tensor.data.numpy() # dim = (115, 43264)

	return train_x, test_x, val_x

# LOAD AND RETURN NUMPY ARRAYS
def load_numpy_arrays():
	train_y = np.load("train_y.npy") # dim = (460, 6)
	test_y = np.load("test_y.npy") # dim = (115, 6)
	val_y = np.load("val_y.npy") # dim = (115, 6)
	return train_y, test_y, val_y

# CONV4 ALEXNET (STOPS AT CONV4 OUTPUT)
class AlexNetConv4(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8] #, 10] # TOOK OUT LAST FEATURES WEIGHT BECAUSE WE ONLY USE UP TO CONV4
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
    def __init__(self,num_classes=6):
        super(AlexNetConv4, self).__init__()
        self.features = nn.Sequential(
            # STOP AT CONV4
            *list(original_model.features.children())[:-3]
            )
        self.load_weights()

    def forward(self, x):
        # INPUT: x is a torch.FloatTensor of size 1x3x227x227
        x = self.features(x)
        # print(x.size()) # OUTPUT: (1L, 256L, 13L, 13L)

        # RESHAPE THE TENSOR TO DIMENSION 1x43264
        x = x.view(x.size(0), 256 * 13 * 13) 
        return x

def get_Conv4_output(data_set):
	first_iter = True
	for img_file in data_set:
		img = "cropped_rgb_227/"+img_file
		im = imread(img)[:,:,:3]
		im = im - np.mean(im.flatten())
		im = im/np.max(np.abs(im.flatten()))
		im = np.rollaxis(im, -1).astype(np.float32)

		# PASS THE IMAGE INTO THE MODEL
		im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
		softmax = nn.Softmax()
		output_tensor = model.forward(im_v)

		if first_iter:
			input_tensor = output_tensor
			first_iter = False
			continue

		input_tensor = torch.cat((input_tensor,output_tensor),0)

	return input_tensor

# PLOT LOSS OVER NUMBER OF EPOCHS
def plot_loss_vs_iterations(loss_data):
    # plt.yscale('log')
    plt.title("Model Loss over Epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss_data[0], loss_data[1], '-b', label="Model Loss")
    plt.legend(loc='best')
    plt.show()

# PLOT LEARNING CURVES OF TRAINING, VALIDATION, AND TEST SETS OVER NUMBER OF EPOCHS
def plot_learning_curves(accuracy_data):
    # plt.yscale('log')
    plt.title("Training, Validation, and Test Set Learning Curves")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracy_data["train_set"][0], accuracy_data["train_set"][1], '-g', label="Train Set")
    plt.plot(accuracy_data["val_set"][0], accuracy_data["val_set"][1], '-b', label="Validation Set")
    plt.plot(accuracy_data["test_set"][0], accuracy_data["test_set"][1], '-r', label="Test Set")
    plt.legend(loc='best')
    plt.show()  

def part10():
	
	# DEFINE THE ALEXNET MODEL
	global original_model, model
	original_model = torchvision.models.alexnet(pretrained=True)
	model = AlexNetConv4()

	# RESIZE THE IMAGES AND PUT THEM INTO FOLDER cropped_rgb_277
	# resize_images("../part8/cropped_rgb_32/", 227)

	# DEFINE THE ACTORS TO BE USED IN create_sets
	actors = ["Bracco", "Gilpin", "Harmon", "Baldwin", "Hader", "Carell"]

	# TRY 32x32 PIXEL IMAGES AND 64x64 PIXEL IMAGES
	folder = "cropped_rgb_227/"

	# PARTITION THE DATA INTO THREE SETS
	training_set, validation_set, test_set = create_sets(folder, actors)

	# OBTAIN THE TARGET VECTORS ASSOCIATED WITH IMAGE SET X (RUN ONLY IF FILES HAVE NOT BEEN CREATED)
	# train_x, train_y = get_x_and_y_data(training_set,folder)
	# test_x, test_y = get_x_and_y_data(test_set,folder)
	# val_x, val_y = get_x_and_y_data(validation_set,folder)

	# SAVE NUMPY ARRAYS SO YOU DON'T HAVE TO RUN get_x_and_y_data AT EVERY ITERATION
	# save_numpy_arrays(train_x, train_y, test_x, test_y, val_x, val_y)

	# OBTAIN THE TENSORS (RUN ONLY IF FILES HAVE NOT BEEN CREATED)
	# train_x_tensor = get_Conv4_output(training_set)
	# test_x_tensor = get_Conv4_output(test_set)
	# val_x_tensor = get_Conv4_output(validation_set)

	# SAVE THE TENSORS
	# save_tensors(train_x_tensor, test_x_tensor, val_x_tensor)

	# LOAD THE TENSORS AS NUMPY ARRAYS
	train_x, test_x, val_x = load_tensors_as_numpys()

	# LOAD THE NUMPY ARRAYS
	train_y, test_y, val_y = load_numpy_arrays()

	# DEFINE ARCHITECTURE PARAMETERS OF NETWORK
	dim_x = 256 * 13 * 13
	dim_h = 45	# 38 is also good
	dim_out = 6

	# DEFINE THE NEURAL NETWORK MODEL
	model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),torch.nn.ReLU(),torch.nn.Linear(dim_h, dim_out),)
	model.eval()
	torch.manual_seed(0)

	# DEFINE A LOSS FUNCTION
	loss_fn = torch.nn.CrossEntropyLoss()

	# TRAIN AND TEST ON ALL SETS
	accuracy_data, loss_data  = train_and_test(model,train_x,train_y,val_x,val_y,test_x,test_y,lr=0.0005, batch_size=45)

	# PLOT THE LOSS OVER ITERATIONS
	plot_loss_vs_iterations(loss_data)

	# PLOT THE LEARNING CURVES
	plot_learning_curves(accuracy_data)


################################################################################################
#________________________________________RUN THE CODE__________________________________________#
################################################################################################
part10()





