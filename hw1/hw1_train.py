import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', encoding = 'big5')
filter1 = train['測項'] == 'PM2.5'
filter2 = train['測項'] == 'O3'
filter3 = train['測項'] == 'SO2'
pm2_5 = train[filter1].ix[:,3:]
o3 = train[filter2].ix[:,3:]
so2 = train[filter3].ix[:,3:]

list_x = []
list_y = []

for i in range(15):
    tempx = pm2_5.iloc[:,i:i+9]        
    tempx.columns = np.array(range(9))
    tempy = pm2_5.iloc[:,i+9]         
    tempy.columns = ['1']
    list_x.append(tempx)
    list_y.append(tempy)
pm2_5_x = pd.concat(list_x, axis = 0)
pm2_5_x = np.array(pm2_5_x, float)
mean = pm2_5_x.mean()

list_x = []

for i in range(15):
    tempx = o3.iloc[:,i:i+9]
    tempx.columns = np.array(range(9))
    list_x.append(tempx)
o3_x = pd.concat(list_x, axis = 0)
o3_x = np.array(o3_x, float)
list_x = []

for i in range(15):
    tempx = so2.iloc[:,i:i+9]
    tempx.columns = np.array(range(9))
    list_x.append(tempx)
so2_x = pd.concat(list_x, axis = 0)
so2_x = np.array(so2_x, float)

x = np.concatenate((pm2_5_x, o3_x, so2_x), axis = 1)    
mean = x.mean()
std = x.std()
x = (x-mean) / std

nor = np.array([mean, std])
pd.DataFrame(nor).to_csv('nor.csv', header = None, index = False)

y = pd.concat(list_y)     
y = np.array(y, float)

x = np.concatenate((np.ones((x.shape[0],1)),x), axis = 1)

w = np.zeros(x.shape[1])


lr = 0.1
 
lr_w = np.zeros(x.shape[1])
for i in range(100000):
    tmp = np.dot(x,w)     
    loss = y - tmp     
    grad = np.dot(x.T,loss)*(-2)
    lr_w += grad**2
    ada = np.sqrt(lr_w)
    w = w - lr*grad/ada
pd.DataFrame(w).to_csv('w.csv', header=None, index=False)

test = pd.read_csv('./test.csv', encoding = 'big5')
pm2_5 = test[test['AMB_TEMP']=='PM2.5'].ix[:,2:]
o3 = test[test['AMB_TEMP']=='O3'].ix[:,2:]
so2 = test[test['AMB_TEMP']=='SO2'].ix[:,2:]
pm2_5 = np.array(pm2_5, float)
o3 = np.array(o3, float)
so2 = np.array(so2, float)

test_x = np.concatenate((pm2_5, o3, so2), axis = 1) 
test_x = (test_x-mean) / std
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis = 1)
ans = np.dot(test_x,w)

pd.DataFrame([['id_' + str(i), ans[i]] for i in range(ans.shape[0])], columns=['id', 'value']) \
          .to_csv('result.csv', index=False)
