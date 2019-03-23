import pandas as pd
import numpy as np
import sys

X_train_path = sys.argv[3]
Y_train_path = sys.argv[4]
X_test_path = sys.argv[5]
output_path = sys.argv[6]

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def sign(x):
    ret = []
    for i in x:
        if i < 0.52:
            ret.append(0)
        else:
            ret.append(1)
    return ret

raw_data = pd.read_csv(X_train_path)
train_x = np.array(raw_data, float)
test_x = pd.read_csv(X_test_path)
test_x = np.array(test_x, float)
train_y = pd.read_csv(Y_train_path)
train_y = np.array(train_y, float)

#normalize
mean = train_x[:, [0,1,3,4,5]].mean(axis=0)
std = train_x[:, [0,1,3,4,5]].std(axis=0)
train_x[:, [0,1,3,4,5]] = (train_x[:, [0,1,3,4,5]]-mean) / std
test_x[:, [0,1,3,4,5]] = (test_x[:, [0,1,3,4,5]]-mean) / std

#shuffle
random = np.arange(train_x.shape[0])
np.random.shuffle(random)
train_x = train_x[random]
train_y = train_y[random]

#initialize
train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)
train_y = train_y.flatten()
seed = 100
mu = 0.0
sigma = 0.01
r_obj = np.random.RandomState(seed)
w = r_obj.normal(loc=mu, scale=sigma, size=train_x.shape[1])
lr_w = np.zeros(train_x.shape[1])
epoch = 10000
lr = 0.01
#logistic regression
for i in range(epoch):
    z = np.dot(train_x, w)
    y = sigmoid(z)    
    loss = train_y - y    
    grad = np.dot(train_x.T,loss)*(-2)
    lr_w += grad**2
    ada = np.sqrt(lr_w)
    w = w - lr*grad/ada

#predict
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
z = np.dot(test_x, w)
test_y = sign(sigmoid(z))          
pd.DataFrame([[str(i+1), test_y[i]] for i in range(len(test_y))], columns=['id', 'label']) \
          .to_csv(output_path, index=False)
