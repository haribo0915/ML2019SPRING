import sys
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
os.environ[ " CUDA_VISIBLE_DEVICES "] = "2"

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # [64, 24, 24]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 3, 3]
        )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
        )

        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

model = Classifier()
model.load_state_dict(torch.load('./model.pth'))
model.cuda()

train_path = sys.argv[1]
out_path = sys.argv[2]
img_row, img_col = 48, 48

raw_data = pd.read_csv(train_path)
x_train = raw_data['feature'].values
x_train = np.array([row.split(' ') for row in x_train], dtype=np.float32)
y_train = raw_data['label'].values
x_train = x_train / 255
x_train = x_train.reshape(x_train.shape[0], 1, img_row, img_col)

idx = 0
x, y = [], []
for i in range(len(y_train)):
	if idx == 7:
		break
	if y_train[i] == idx:
		x.append(x_train[i])
		y.append(y_train[i])
		idx = idx + 1
x = np.array(x)
y = np.array(y)
x = torch.FloatTensor(x)
y = torch.LongTensor(y)


def compute_saliency_maps(x, y, model):
    model.eval()
    x.requires_grad_()
    y_pred = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliency = x.grad.abs().squeeze().data
    return saliency
def show_saliency_maps(x, y, model):
    x_org = x.squeeze().numpy()
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(x, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().cpu().numpy()
    
    num_pics = x_org.shape[0]
    for i in range(num_pics):
        # You need to save as the correct fig names
        plt.imsave(out_path + 'fig1_'+ str(i) + '.jpg', saliency[i], cmap=plt.cm.jet)

show_saliency_maps(x, y, model)    