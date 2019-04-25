import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd.gradcheck import zero_gradients
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torchvision.models as models
from torchvision import transforms, utils
import csv
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, \
                               resnet101, densenet121, densenet169

import sys
import os

img_dir = sys.argv[1]
out_dir = sys.argv[2]

CLIP_MAX = 1.0
CLIP_MIN = 0.0

class Attacker:
    def __init__(self, clip_max=CLIP_MAX, clip_min=CLIP_MIN):
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        pass
class BIM(Attacker):
    def __init__(self, eps=0.1, eps_iter=0.01, n_iter=30, clip_max=CLIP_MAX, clip_min=CLIP_MIN):
        super(BIM, self).__init__(clip_max, clip_min)
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_iter = n_iter

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        ny = y
        nx.requires_grad = True
        eta = torch.zeros(nx.shape).to(device)

        for i in range(self.n_iter):
            out = model(nx+eta)
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps_iter * torch.sign(nx.grad.data)
            eta = torch.clamp(eta, -self.eps, self.eps)
            nx.grad.data.zero_()

        x_adv = nx + eta
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        x_adv = x_adv.squeeze(0)
        
        return x_adv.cpu().detach()

def Reading_TrueLabels(path):
    rows = list(csv.reader(open(path, 'r')))
    TrueLabel = []
    for row in rows[1::]:
        TrueLabel.append(int(row[3]))
    TrueLabel = np.array(TrueLabel).reshape(200, 1)
    return TrueLabel

def Reading_Categories(path):
    rows = list(csv.reader(open(path, 'r')))
    Name = []
    for row in rows[1::]:
        tmp = row[1].split(',')
        Name.append(tmp[0])
    Name = np.array(Name).reshape(1000, 1)
    return Name

if __name__ == '__main__':
    pretrained_model = models.vgg19(pretrained=True)
    use_cuda=True

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model = pretrained_model.to(device)
    model.eval()

    '''pretrained_model2 = models.resnet101(pretrained=True)
    model2 = pretrained_model2.to(device)
    model2.eval()'''

    i1 = 0; j1 = 0; k1 = 0; correct = 0

    target = Reading_TrueLabels('labels.csv')
    target = torch.LongTensor(target)

    Categories = Reading_Categories('categories.csv')

    attacker = BIM(eps=0.0195, eps_iter=0.008, n_iter=10, clip_max=CLIP_MAX, clip_min=CLIP_MIN)
    for i in range(200):
        trans = transforms.Compose([transforms.ToTensor()]) #range[0, 255] -> range[0.0, 1.0]

        img_path = os.path.join(img_dir, str(i1) + str(j1) + str(k1) + '.png')
        tmp1 = Image.open(img_path).convert('RGB')
        image = trans(tmp1)
        tmp2 = image
            
        image = attacker.generate(model, image.to(device), target[i].to(device))

        '''diff = np.array(tmp2-image, dtype=float)
        diff = np.abs(diff)
        print('L-inf: %f' % (np.max(diff)))'''

        tmp = torch.unsqueeze(image, 0)
        output = model(tmp.to(device))

        '''matplotlib.rc('xtick', labelsize=12)
        if i % 51 == 3:
            tmp = torch.unsqueeze(tmp2, 0)
            output_ori = model(tmp.to(device))

            #plot adversarial
            top = output[0].topk(3) #2-dim first is prob, second is index 

            objects = (str(Categories[top[1][0]][0])+'('+str(top[1][0].item())+')', \
                       str(Categories[top[1][1]][0])+'('+str(top[1][1].item())+')', \
                       str(Categories[top[1][2]][0])+'('+str(top[1][2].item())+')')

            y_pos = np.arange(len(objects))
            performance = [top[0][0], top[0][1], top[0][2]]
             
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.yticks(np.arange(0, 110, 10))
            plt.title('Adversarial image')
             
            plt.savefig('Adversarial_'+str(i)+'.png')
            plt.clf()
            #plot origin
            top_ori = output_ori[0].topk(3)

            objects_ori = (str(Categories[top_ori[1][0]][0])+'('+str(top_ori[1][0].item())+')', \
                       str(Categories[top_ori[1][1]][0])+'('+str(top_ori[1][1].item())+')', \
                       str(Categories[top_ori[1][2]][0])+'('+str(top_ori[1][2].item())+')')

            performance_ori = [top_ori[0][0], top_ori[0][1], top_ori[0][2]]

            plt.bar(y_pos, performance_ori, align='center', alpha=0.5)
            plt.xticks(y_pos, objects_ori)
            plt.yticks(np.arange(0, 110, 10))
            plt.title('Original image')
             
            plt.savefig('Origin_'+str(i)+'.png')
            plt.clf()'''
        

        final_pred = output.max(1, keepdim=True)[1]
        print(final_pred.item(), target[i].item(), sep=' ')
        if final_pred.item() == target[i].item():
            correct += 1

        image = transforms.ToPILImage()(image).convert('RGB')

        out_path = os.path.join(out_dir, str(i1) + str(j1) + str(k1) + '.png')
        image.save(out_path)
        
        k1 += 1
        if k1 == 10:
            k1 = 0
            j1 += 1
        if j1 == 10:
            j1 = 0
            i1 += 1

    final_acc = correct/200.0
    print("Test Accuracy = {} / 200 = {}".format(correct, final_acc))        