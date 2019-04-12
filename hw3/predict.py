import numpy as np
import pandas as pd
import sys
from keras.models import load_model

test_path = sys.argv[1]
output_path = sys.argv[2]

model = [None]*4
model[0] = load_model('model1.h5')
model[1] = load_model('model2.h5')
model[2] = load_model('model3.h5')
model[3] = load_model('model4.h5')

raw = pd.read_csv(test_path)
x_test = raw['feature'].values
x_test = np.array([row.split(' ') for row in x_test], dtype=np.float32)

x_test /= 255
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

y_test = model[0].predict(x_test)
for i in range(1, 4):
	y_test += model[i].predict(x_test)
y = np.zeros(y_test.shape[0])
for i in range(y_test.shape[0]):
    y[i] = int(np.argmax(y_test[i]))


pd.DataFrame([[i, int(y[i])] for i in range(y.shape[0])], columns=['id', 'label']) \
        .to_csv(output_path, index=False)