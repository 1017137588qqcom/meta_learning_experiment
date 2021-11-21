from torchvision import datasets, transforms
import torch
import os
import numpy as np
from torch.autograd import Variable

if not os.path.isdir('./data/multi_tasks'):
    os.mkdir('./data/multi_tasks')

metaTraintoy_data_lst = []
metaValToy_data_lst = []
for i in range(50):
    [mean]= np.random.uniform(0, 5, 1)
    [std] = np.random.uniform(0, 5, 1)
    print('Meta_training', i, mean, std)
    metaTraintoy_data_lst.append(Variable(torch.normal(mean, std, (10000, 5))))
    metaValToy_data_lst.append(Variable(torch.normal(mean, std, (10000, 5))))

metaTraintoy_data = torch.cat(metaTraintoy_data_lst, 0)
metaValToy_data = torch.cat(metaValToy_data_lst, 0)
print('Meta_training dataset', metaTraintoy_data.shape, metaValToy_data.shape)

metaTestTraintoy_data_lst = []
metaTestValToy_data_lst = []
for i in range(50):
    [mean]= np.random.uniform(0, 5, 1)
    [std] = np.random.uniform(0, 5, 1)
    print('Meta_testing', i, mean, std)
    metaTestTraintoy_data_lst.append(Variable(torch.normal(mean, std, (10000, 5))))
    metaTestValToy_data_lst.append(Variable(torch.normal(mean, std, (10000, 5))))
metaTest_train = torch.cat(metaTestTraintoy_data_lst, 0)
metaTest_val = torch.cat(metaTestValToy_data_lst, 0)
print('Meta_testing dataset', metaTraintoy_data.shape, metaValToy_data.shape)

torch.save(metaTraintoy_data, './data/multi_tasks/TrainData.pth')
torch.save(metaValToy_data,   './data/multi_tasks/valData.pth')

torch.save(metaTest_train,  './data/multi_tasks/testTrainData.pth')
torch.save(metaTest_val,    './data/multi_tasks/testValData.pth')