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

# ORIGINAL ALEXNET (ALL LAYERS)
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# CONV4 ALEXNET (STOPS AT CONV4 OUTPUT)
class AlexNetConv4(nn.Module):
    def __init__(self):
        super(AlexNetConv4, self).__init__()
        self.features = nn.Sequential(
            # STOP AT CONV4
            *list(original_model.features.children())[:-3]
            )
    def forward(self, x):
        x = self.features(x) # try only having this line
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
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
		# all_probs = softmax(model.forward(im_v)).data.numpy()[0]
		# sorted_ans = np.argsort(all_probs)
	return input_tensor

def part10():
	# resize_images("../part8/cropped_rgb_32/", 227)

	actors = ["Bracco", "Gilpin", "Harmon", "Baldwin", "Hader", "Carell"]

	# TRY 32x32 PIXEL IMAGES AND 64x64 PIXEL IMAGES
	folder = "cropped_rgb_227/"

	# PARTITION THE DATA INTO THREE SETS
	training_set, validation_set, test_set = create_sets(folder, actors)

	# OBTAIN THE TARGET VECTORS ASSOCIATED WITH IMAGE SET X
	# train_x, train_y = get_x_and_y_data(training_set,folder)
	# test_x, test_y = get_x_and_y_data(test_set,folder)
	# val_x, val_y = get_x_and_y_data(validation_set,folder)

	# ORIGINAL ALEXNET (ALL LAYERS)
	# model_orig = torchvision.models.alexnet(pretrained=True)
	# model = MyAlexNet()
	# model.eval()
	
	global original_model
	original_model = torchvision.models.alexnet(pretrained=True)

	global model
	model = AlexNetConv4()

	# train_x = get_Conv4_output(training_set)

	# SAVE THE TENSOR SO YOU DON'T HAVE TO RETREIVE IT EVERY ITERATION
	# torch.save(train_x, "train_x.pth")

	x = torch.load("train_x.pth")
	print(x.shape) # OUTPUTS: (460L, 256L, 13L, 13L)

	# DEFINE ARCHITECTURE PARAMETERS OF NETWORK
	dim_x = 460
	dim_h = 45	# 38 is also good
	dim_out = 6

	# DEFINE THE NEURAL NETWORK MODEL
	# model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),torch.nn.ReLU(),torch.nn.Linear(dim_h, dim_out),)
	# model.eval()
	# model.apply(init_weights)
	# torch.manual_seed(0)
	# torch.manual_seed_all(0)

	# y_pred = model(x)

	# DEFINE A LOSS FUNCTION
	loss_fn = torch.nn.CrossEntropyLoss()


	# all_probs = softmax(model.forward(im_v)).data.numpy()[0]
	# sorted_ans = np.argsort(all_probs)


################################################################################################
#________________________________________RUN THE CODE__________________________________________#
################################################################################################
part10()









