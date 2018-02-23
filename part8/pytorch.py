from torch.autograd import Variable
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

#%matplotlib inline  

M = loadmat("mnist_all.mat")


def get_test(M):
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))
    
    test_k =  ["test"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:])/255.)))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[test_k[k]]), 1))))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))
    
    train_k =  ["train"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:])/255.)  ))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        

train_x, train_y = get_train(M)
test_x, test_y = get_test(M)

# print(train_x.shape, train_y.shape)
# print(test_x.shape, test_y.shape)

dim_x = 28*28
dim_h = 150
dim_out = 10

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor



################################################################################
#Subsample the training set for faster training

train_idx = np.random.permutation(range(train_x.shape[0]))[:1300]
x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)
#################################################################################

# DEFINE THE NEURAL NETWORK
model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),)

# DEFINE A LOSS FUNCTION
loss_fn = torch.nn.CrossEntropyLoss()

# TRAIN THE MODEL USING ADAM, A VARIANT OF GRADIENT DESCENT
learning_rate = 0.05
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
iterations = 10000

def test_model(x_set, y_set):
    # MAKE PREDICTIONS FOR THE SET DATA
    x = Variable(torch.from_numpy(x_set), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()

    # LOOK AT THE PERFORMANCE
    accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(y_set, 1))
    return accuracy

accuracy_history = []
for t in range(iterations):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step
    if t%200 == 0:
            print("Optimization is {}% complete.".format(100*t/iterations))
            print("Accuracy: {}".format(test_model(train_x, train_y)))
            started = True
            accuracy_history.append(test_model(train_x,train_y))
            # if len(accuracy_history)>1 and abs(accuracy_history[-1]-accuracy_history[-2]) < 0.001:
            #     learning_rate /= 2
            #     print("Adjusted the learning rate.")
            #     print("Learning rate is now: ", learning_rate)
                
# MAKE PREDICTIONS FOR THE TEST SET
x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
y_pred = model(x).data.numpy()
# print(y_pred[0],y_pred[2], y_pred[8])

# LOOK AT THE PERFORMANCE
np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))

# YOU CAN ACCESS THE WEIGHTS LIKE THIS
model[0].weight

# LOOK AT THE WEIGHTS ASSOCIATED WITH UNIT 10
model[0].weight.data.numpy()[10, :].shape

# PLOT THE WEIGHTS ASSOCIATED WITH UNIT 10
# plt.imshow(model[0].weight.data.numpy()[10, :].reshape((28, 28)), cmap=plt.cm.coolwarm)
# plt.show()

# PLOT THE WEIGHTS ASSOCIATED WITH UNIT 12
# plt.imshow(model[0].weight.data.numpy()[12, :].reshape((28, 28)), cmap=plt.cm.coolwarm)
# plt.show()

