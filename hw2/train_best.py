import pandas as pd
import numpy as np
import sys

from sklearn.tree import DecisionTreeClassifier

X_train_path = sys.argv[3]
Y_train_path = sys.argv[4]
X_test_path = sys.argv[5]
output_path = sys.argv[6]

def Adaboost_clf(train_x, train_y, test_x, iter, clf):
    # Initialize weights
    w = np.ones(train_x.shape[0])
    pred_test = np.zeros(test_x.shape[0])
    for i in range(iter):
        clf.fit(train_x, train_y, sample_weight = w)
        pred_train_i = clf.predict(train_x)
        miss = [int(x) for x in (pred_train_i != train_y)]
        miss2 = [x if x==1 else -1 for x in miss]
        err_m = np.dot(w,miss) / sum(w)
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        pred_test_i = clf.predict(test_x)
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha_m for x in pred_test_i])]
    pred_test = np.sign(pred_test)
    return pred_test

raw_data = pd.read_csv(X_train_path, header=0) 
train_x = raw_data.values
test_x = pd.read_csv(X_test_path, header=0)
test_x = test_x.values
train_y = pd.read_csv(Y_train_path, header=0)
train_y = train_y.values.flatten()
for i in range(len(train_y)):
    if train_y[i] == 0:
        train_y[i] = -1

clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
test_y = Adaboost_clf(train_x, train_y, test_x, 400, clf_tree).astype(int)

for i in range(len(test_y)):
    if test_y[i] == -1:
        test_y[i] = 0
pd.DataFrame([[str(i+1), test_y[i]] for i in range(len(test_y))], columns=['id', 'label']) \
          .to_csv(output_path, index=False)




