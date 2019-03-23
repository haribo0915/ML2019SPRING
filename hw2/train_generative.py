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
        if i > 0.52:
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
train_x_1 = train_x[train_y[:, 0] == 0, :]
train_x_2 = train_x[train_y[:, 0] == 1, :]
n1 = train_x_1.shape[0]
n2 = train_x_2.shape[0]
u1 = np.mean(train_x_1 ,axis=0)
u2 = np.mean(train_x_2 ,axis=0)
p1 = n1/(n1+n2)
p2 = 1 - p1
cov =  np.cov(train_x_1,rowvar=False) * p1 + np.cov(train_x_2,rowvar=False) * p2

cov_inv = np.linalg.pinv(cov)

#generative model
z =((test_x).dot(cov_inv).dot(u1-u2)- 
    (1/2)*(u1).dot(cov_inv).dot(u1)+ (1/2)*(u2).dot(cov_inv).dot(u2)
    +np.log(n1/n2))

test_y = sign(sigmoid(z))        
pd.DataFrame([[str(i+1), test_y[i]] for i in range(len(test_y))], columns=['id', 'label']) \
          .to_csv(output_path, index=False)
