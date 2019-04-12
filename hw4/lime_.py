import sys
import torch.nn as nn
import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import slic
from skimage import io, data, color
import matplotlib.pyplot as plt

out_path = sys.argv[2]

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

# load data and model
def readfile(path):
    print("Reading File...")
    x_train = []
    x_label = []
    val_data = []
    val_label = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(48, 48)
        x_train.append(tmp)
        x_label.append(raw_train[i][0])

    x_train = np.array(x_train, dtype=np.float32) / 255.0
    x_label = np.array(x_label, dtype=int)

    return x_train, x_label

model = Classifier().cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval().cuda()

# two functions that lime image explainer requires
def predict(input_):
    # Input: image tensor
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    # ex: return model(data).numpy()
    input_ = np.array(input_, dtype=np.float32)
    x = []
    for i in range(len(input_)):
        x.append(color.rgb2gray(input_[i]).reshape(1, 48, 48))
    x = np.array(x, dtype=np.float32)
    x = torch.FloatTensor(x)
    y = model(x.cuda()).detach().cpu().numpy()
    return y
    

def segmentation(input_):
    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    # ex: return skimage.segmentation.slic()
    segments = slic(input_, n_segments=60, compactness=10)
    return segments

x_train, x_label = readfile(sys.argv[1])

idx = 0
x, y = [], []
for i in range(len(x_label)):
    if idx == 7:
        break
    if x_label[i] == idx:
        x.append(x_train[i])
        y.append(x_label[i])
        idx = idx + 1
x = np.array(x)
y = np.array(y)

# Lime needs RGB images
x_train_rgb = []
for i in range(len(x)):
    tmp = color.gray2rgb(x[i])
    x_train_rgb.append(tmp)

x_train_rgb = np.array(x_train_rgb, dtype=np.float32)

for i in range(len(x_train_rgb)):
    # Initiate explainer instance
    explainer = lime_image.LimeImageExplainer()
    np.random.seed(16)
    # Get the explaination of an image
    explaination = explainer.explain_instance(
                                image=x_train_rgb[i], 
                                classifier_fn=predict,
                                segmentation_fn=segmentation
                            )

    # Get processed image
    image, mask = explaination.get_image_and_mask(
                                    label=y[i],
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0
                                )

    # save the image
    plt.imsave(out_path+'fig3_'+str(i)+'.jpg', image)