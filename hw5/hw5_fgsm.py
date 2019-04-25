import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd.gradcheck import zero_gradients
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
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

def Reading_TrueLabels(path):
	rows = list(csv.reader(open(path, 'r')))
	TrueLabel = []
	for row in rows[1::]:
		TrueLabel.append(int(row[3]))
	TrueLabel = np.array(TrueLabel).reshape(200, 1)
	return TrueLabel

if __name__ == '__main__':
	pretrained_model = models.resnet50(pretrained=True)
	use_cuda=True

	device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

	model = pretrained_model.to(device)
	model.eval()

	i1 = 0; j1 = 0; k1 = 0; epsilon = 0.07

	target = Reading_TrueLabels('labels.csv')
	target = torch.LongTensor(target)

	for i in range(200):
		trans = transforms.Compose([transforms.ToTensor()]) #range[0, 255] -> range[0.0, 1.0]

		img_path = os.path.join(img_dir, str(i1) + str(j1) + str(k1) + '.png')
		tmp = Image.open(img_path).convert('RGB')
		image = trans(tmp)
			
		image = image.unsqueeze(0)
		image = image.to(device)
		image.requires_grad = True
			    
		# set gradients to zero
		zero_gradients(image)
		
		output = model(image)
		loss = F.cross_entropy(output, target[i].to(device))
		loss.backward() 
			    
		# add epsilon to image
		image = image + epsilon * image.grad.sign()
		image = torch.clamp(image, CLIP_MIN, CLIP_MAX)
		image = image.squeeze(0)
		image = image.cpu().detach()

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