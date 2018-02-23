from pylab import *
import numpy as np
from torch.autograd import Variable
import torch
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


act = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))    
testfile = urllib.URLopener()
image_files = []
photo_num = dict([("Bracco",0), ("Gilpin",0), ("Harmon",0), ("Baldwin",0), ("Hader",0), ("Carell",0), ("Radcliffe",0), ("Butler",0), ("Vartan",0), ("Chenoweth",0), ("Drescher",0), ("Ferrera",0)])                                                                       


################################################################################################
#______________________________________PART 1 FUNCTIONS________________________________________#
################################################################################################

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

def resize_images(folder):
    for file in os.listdir(folder):
        # fillename = str(file)
        if file != '.DS_Store':
            small_res_matrix = imread(folder + file)
            large_res_matrix = imresize(small_res_matrix, (64,64))
            mpimg.imsave("cropped_rgb_64/"+file, large_res_matrix) # save the image in another folder

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
    cg_matrix = c_matrix
    cgr_matrix = imresize(cg_matrix, (32,32)) # cropped grayscale resized matrix
    
    photo_num[actor_name[1]] += 1 # add one to the actor's key every time they get a photo saved
    # print("saving:" + filename)
    # print("WHILE SAVING:")
    # print("size:" +str(cgr_matrix.size))
    # print("shape:" + str(cgr_matrix.shape))
    # print("image matrix of"+str(filename) + "is" + str(cgr_matrix))

    mpimg.imsave("cropped_rgb_64/"+filename, cgr_matrix, cmap=plt.cm.gray) # save the image
    
    #TO SHOW AN IMAGE (IN GRAY SCALE) FROM ITS IMAGE MATRIX USE:
    #matplotlib.pyplot.imshow(cg_matrix,cmap=plt.cm.gray)
    #plt.show()

################################################################################################
#______________________________________PART 2 FUNCTIONS________________________________________#
################################################################################################

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

# SUBSAMPLE THE TRAINING SET FOR FASTER TRAINING
def subsample(train_x,train_y,batch_size):
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    train_idx = np.random.permutation(range(train_x.shape[0]))[:batch_size] # CHANGED THIS FROM 1000
    x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)
    return x, y_classes

# INITIALIZE WEIGHTS
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.000001)

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
    iterations = 10000
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
        if t%500 == 0:
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

    if started: print("Optimization is 100% complete.")

    return accuracy_data, loss_data

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

# PLOT ACCURACY WHILE VARYING A PARTICULAR PARAMETER
def plot_accuracy_with_parameter(param_data,test_data,parameter):
    # plt.yscale('log')
    plt.title("Accuracy of Neural Network with Varying " + parameter)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for i in range(len(test_data)):
        plt.plot(test_data[i][0], test_data[i][1], label=str(param_data[i]))
    plt.legend(loc='best')
    plt.show()  

# SHOWS DEPENDENCY OF ACCURACY ON BATCH SIZE
def compare_batch_size_performance(train_x,train_y,val_x,val_y,test_x,test_y):
    # DEFINE ARCHITECTURE PARAMETERS OF NETWORK
    dim_x = 32*32 + 1
    dim_out = 6
    dim_h = 45

    batch_size_data = []
    test_data = []
    for batch_size in range(15,50,5):
        # DEFINE THE NEURAL NETWORK MODEL
        model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),torch.nn.ReLU(),torch.nn.Linear(dim_h, dim_out),)
        model.eval()

        model.apply(init_weights)
        torch.manual_seed(0)

        # DEFINE A LOSS FUNCTION
        loss_fn = torch.nn.CrossEntropyLoss()

        # TRAIN AND TEST ON ALL SETS
        accuracy_data, loss_data  = train_and_test(model,train_x,train_y,val_x,val_y,test_x,test_y,batch_size=batch_size)
        
        # COLLECT DATA TO PLOT
        batch_size_data.append(batch_size)
        test_data.append(accuracy_data["test_set"])

    plot_accuracy_with_parameter(batch_size_data, test_data, "batch size")

# SHOWS DEPENDENCY OF ACCURACY ON dim_h
def compare_dim_h_performance(train_x,train_y,val_x,val_y,test_x,test_y):
    # DEFINE ARCHITECTURE PARAMETERS OF NETWORK
    dim_x = 32*32 + 1
    dim_out = 6

    dim_h_data = []
    test_data = []
    for dim_h in range(35,60,2):
        # DEFINE THE NEURAL NETWORK MODEL
        model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),torch.nn.ReLU(),torch.nn.Linear(dim_h, dim_out),)
        model.eval()

        model.apply(init_weights)
        torch.manual_seed(0)

        # DEFINE A LOSS FUNCTION
        loss_fn = torch.nn.CrossEntropyLoss()

        # TRAIN AND TEST ON ALL SETS
        accuracy_data, loss_data  = train_and_test(model,train_x,train_y,val_x,val_y,test_x,test_y)
        
        # COLLECT DATA TO PLOT
        dim_h_data.append(dim_h)
        test_data.append(accuracy_data["test_set"])

    plot_accuracy_with_parameter(dim_h_data, test_data, "dim_h")

def compare_lr_performance(train_x,train_y,val_x,val_y,test_x,test_y):
    # DEFINE ARCHITECTURE PARAMETERS OF NETWORK
    dim_x = 32*32 + 1
    dim_out = 6
    dim_h = 45
    lr_data = []
    test_data = []

    for lr in [0.0001,0.0003,0.0005,0.0007,0.0008]: # more general: [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]:
        # DEFINE THE NEURAL NETWORK MODEL
        model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),torch.nn.ReLU(),torch.nn.Linear(dim_h, dim_out),)
        model.eval()

        model.apply(init_weights)
        torch.manual_seed(0)

        # DEFINE A LOSS FUNCTION
        loss_fn = torch.nn.CrossEntropyLoss()

        # TRAIN AND TEST ON ALL SETS
        accuracy_data, loss_data  = train_and_test(model,train_x,train_y,val_x,val_y,test_x,test_y,lr=lr)
        
        # COLLECT DATA TO PLOT
        lr_data.append(lr)
        test_data.append(accuracy_data["test_set"])

    plot_accuracy_with_parameter(lr_data, test_data, "learning_rate")

################################################################################################
#_______________________________________TEST FUNCTIONS_________________________________________#
################################################################################################
def part8a():
    print("_________________________________________________________")
    print("PART1: DOWNLOADING JPGS FROM TEXT (NO OUTPUT)")
    print("_________________________________________________________")
    # download_jpgs_from_txt("subset_actors.txt")
    resize_images("cropped_rgb/")

def part8b():
    print("_________________________________________________________")
    print("PART2: PARTIONIONING ACTORS INFO INTO SETS (NO OUTPUT)")
    print("_________________________________________________________")
    actors = ["Bracco", "Gilpin", "Harmon", "Baldwin", "Hader", "Carell"]

    # TRY 32x32 PIXEL IMAGES AND 64x64 PIXEL IMAGES
    folder = "cropped_rgb_32/"
    # folder = "cropped_rgb_64/"

    # PARTITION THE DATA INTO THREE SETS
    training_set, validation_set, test_set = create_sets(folder, actors)

    # OBTAIN THE TARGET VECTORS ASSOCIATED WITH IMAGE SET X
    train_x, train_y = get_x_and_y_data(training_set,folder)
    test_x, test_y = get_x_and_y_data(test_set,folder)
    val_x, val_y = get_x_and_y_data(validation_set,folder)

    # DEFINE ARCHITECTURE PARAMETERS OF NETWORK
    dim_x = 32*32 + 1 
    dim_h = 45	# 38 is also good
    dim_out = 6

    # DEFINE THE NEURAL NETWORK MODEL
    model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),torch.nn.ReLU(),torch.nn.Linear(dim_h, dim_out),)
    model.eval()
    model.apply(init_weights)
    torch.manual_seed(0)
    # torch.manual_seed_all(0)
    
    # DEFINE A LOSS FUNCTION
    loss_fn = torch.nn.CrossEntropyLoss()

    # TRAIN AND TEST ON ALL SETS
    accuracy_data, loss_data  = train_and_test(model,train_x,train_y,val_x,val_y,test_x,test_y)

    # PLOT THE LOSS OVER ITERATIONS
    # plot_loss_vs_iterations(loss_data)

    # PLOT THE LEARNING CURVES
    # plot_learning_curves(accuracy_data)

    # PLOT THE LEARNING CURVES WHILE VARYING PARAMETER VALUES
    # 1. dim_h
    # compare_dim_h_performance(train_x,train_y,val_x,val_y,test_x,test_y)
    # 2. batch_size
    # compare_batch_size_performance(train_x,train_y,val_x,val_y,test_x,test_y)
    # 3. learning_rate
    # compare_lr_performance(train_x,train_y,val_x,val_y,test_x,test_y)

    # SAVE THE MODEL AFTER TRAINING
    torch.save(model, "model.pth") 

def part9():

    # LOAD THE MODEL TRAINED IN PART8b
    model = torch.load('model.pth')

    # # YOU CAN ACCESS THE WEIGHTS LIKE THIS
    # model[0].weight

    # HAVE TO GET RID OF THE BIAS ELEMENT (i.e., convert from dim 1025 to dim 1024)
    model_array = model[0].weight.data.numpy()
    model_array = model_array[:,:-1]
    model_array_out = model[2].weight.data.numpy()

    for unit in range(0,45):
        index = 2
        label = "Harmon"
        if argmax(model_array_out[:,unit]) == 2:
            model_unit_weights = model_array[unit, :]
            plt.imshow(model_unit_weights.reshape((32, 32)), cmap=plt.cm.coolwarm) # CHANGED THIS FROM 28,28
            plt.title('Hidden unit #{} associated with {}'.format(unit,label))
            plt.show()

    # # PLOT THE WEIGHTS ASSOCIATED WITH UNIT 10
    # model_unit_weights = model_array[unit, :]
    # plt.imshow(model_unit_weights.reshape((32, 32)), cmap=plt.cm.coolwarm) # CHANGED THIS FROM 28,28
    # plt.title('Hidden unit #{}'.format(unit))
    # plt.show()

    # UNITS WITH FACES: 3, 4, 6, 7, 13, 16, 19, 20, 21, 25, 27 (Baldwin), 28, 31, 33, 34, 40, 41, 42


    # PLOT THE WEIGHTS ASSOCIATED WITH UNIT 20
    # model_unit_weights = model_array[20, :]
    # plt.imshow(model_unit_weights.reshape((32, 32)), cmap=plt.cm.coolwarm) # CHANGED THIS FROM 28,28
    # plt.show()

################################################################################################
#________________________________________RUN THE CODE__________________________________________#
################################################################################################

# part8a()
# part8b()
# part9()


