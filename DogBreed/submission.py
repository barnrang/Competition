import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from PIL import Image
import os

def load_image(place):
    image = Image.open(place)
    image = test_transform(image)
    return image

def load_image_batch(model,place, file_list, batch_size=100):
    batch_count = 0
    loops = np.int(np.ceil(len(file_list)/batch_size))
    inputs = Variable(torch.zeros(batch_size,3,224,224))
    model.train(False)
    m = nn.Softmax()
    for i in range(loops):
        st = batch_size*i
        ed = st + batch_size
        for index, file_name in enumerate(file_list[st:ed]):
            inputs[index] = load_image(os.path.join(place, file_name))
        inputs = inputs.cuda()
        outputs = model(inputs)
        df_out.iloc[st:ed,1:121] = m(outputs).data.cpu().numpy()

file_list = os.listdir('test')
file_list.sort()
df_out = pd.read_csv('sample_submission.csv')
test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model_conv = models.resnet152()
for param in model_conv.parameters():
    param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1000),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000,500),
    nn.ReLU(),
    nn.Dropout(p=0.75),
    nn.Linear(500, 120)
)

model_conv.load_state_dict(torch.load('model/test3'))
load_image_batch(model_conv,'test',file_list, batch_size=200)
df_out.to_csv('submit-tsubame.csv')
