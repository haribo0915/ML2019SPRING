import pandas as pd
import numpy as np
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

test = pd.read_csv(input_path, encoding = 'big5')
nor = pd.read_csv('nor.csv', header = None)
nor = np.array(nor)
mean = nor[0][0]
std = nor[1][0]

pm2_5 = test[test['AMB_TEMP']=='PM2.5'].ix[:,2:]
o3 = test[test['AMB_TEMP']=='O3'].ix[:,2:]
so2 = test[test['AMB_TEMP']=='SO2'].ix[:,2:]
pm2_5 = np.array(pm2_5, float)
o3 = np.array(o3, float)
so2 = np.array(so2, float)

test_x = np.concatenate((pm2_5, o3, so2), axis = 1) 
test_x = (test_x-mean) / std
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis = 1)

w = pd.read_csv('w.csv', header=None)
w = np.array(w)

ans = np.dot(test_x,w)

pd.DataFrame([['id_'+str(i), ans[i][0]] for i in range(ans.shape[0])],
             columns=['id', 'value']).to_csv(output_path, index=False)