import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from DataSplit import MultiToyTasks
from META_OPTIMIZER_WITHOUT_GRAD_v3 import MetaOptimizer
from utils import get_flat_parameters, set_parameters
from model import ToyModel
print(ToyModel())
import torch.nn.functional as F
import numpy as np
import os
from copy import deepcopy
from threading import Thread, BoundedSemaphore
# from concurrent.futures import ThreadPoolExecutor
import argparse

parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=500, type=int, help='batch size')
parser.add_argument('--val_num', default=10000, type=int, help='batch size')
parser.add_argument('--max_epoch', default=10, type=int, help='max epoch')
parser.add_argument('--outer_loop', default=1000000, type=int, help='outer loop')
parser.add_argument('--inner_loop', default=100, type=int, help='inner loop')
parser.add_argument('--flod', '-f', default=1, type=int, help='test flod')
parser.add_argument('--resume', default='True', help='resume from checkpoint')
args = parser.parse_args()
gpu = "0, 1"           # which GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr_loss = 1000
best_acc = 0        # best test accuracy
start_epoch = 0     # start from epoch 0 or last checkpoint epoch
flod = args.flod
work = True
weight_decay = 0.000
dict_fitness = {}
dict_acc = {}
init_lr = 1e-1
NUM_PARENTS = 25
NUM_KIDS = 25
avg_s = 3
maxjob = BoundedSemaphore(25)

def make_kids(pop, num_kids, DNA_SIZE):
    kids = {'DNA':torch.zeros(num_kids, DNA_SIZE),
            'MUT_STRENGTH':torch.zeros(num_kids, DNA_SIZE)}
    for kv, ks in zip(kids['DNA'], kids['MUT_STRENGTH']):
        ### 随机选两个父代
        p1, p2 = np.random.choice(np.arange(NUM_PARENTS), size=2, replace=True)
        ### 随机生成交叉点
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)
        ### 交叉
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp]= pop['DNA'][p2, ~cp]
        ks[cp] = pop['MUT_STRENGTH'][p1, cp]
        ks[~cp]= pop['MUT_STRENGTH'][p2, ~cp]
        ### 变异
        ks[:] = ks * torch.exp(((2 * (DNA_SIZE**0.5))**(-0.5))*torch.randn(*ks.shape)
                               + (2*DNA_SIZE)**(-0.5)*torch.randn(*ks.shape))
        kv += ks * torch.randn(*kv.shape)
    ### 混合父代和子代
    for key in ['DNA', 'MUT_STRENGTH']:
        pop[key] = torch.cat((pop[key], kids[key]),0)
    return pop

def train(dataloader, net, optimizer, criterion, dvc):
    """Train the network"""
    net.train()
    for num_task in range(args.max_epoch):
        for batch_id, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(dvc), targets.to(dvc)
            outputs = net(inputs)
            loss = criterion(outputs, targets) # .long()
            loss.backward()
            optimizer.step()

def test(dataloader, net, gt, worse_fitness, par, dvc):
    """Validation and the test."""
    net.eval()
    num_id = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(dataloader):
            num_id += 1
            inputs, targets = inputs.to(dvc), targets.to(dvc)
            outputs = net(inputs)
            gt_out  = gt(inputs)
            test_loss = -torch.mean((F.softmax(outputs) - F.softmax(gt_out))**2)
            # sorted_loss, idc = torch.sort(test_loss)
            # min_ = sorted_loss[0]

            _, predicted = outputs.max(1)  # judge max elements in predicted`s Row(1:Row     0:Column)
            correct += predicted.eq(targets).float().sum()  # judge how many elements same in predicted and targets

    fitness = test_loss
    if worse_fitness > fitness:
        worse_fitness = fitness
        dict_acc[par] = (100. * correct / targets.size(0)).view(1, 1).cpu()

    return worse_fitness

def ES(criterion, par):
    maxjob.acquire()
    worse_fitness = 10
    if par % 2 == 0:
        dvc = 'cuda:0'
    else:
        dvc = 'cuda:1'
    for each in range(avg_s):
        net = ToyModel()
        opt = optim.SGD(net.parameters(), lr=init_lr)
        datas = MultiToyTasks(train_batchsize=args.batch_size, val_batchsize=args.val_num)
        top = 50000
        down = 0
        while abs(top - down) >= 5000:
            toy_traindata, toy_valdata = datas.minibatch_train_sampler(num_batch=args.inner_loop)
            gT_model = ToyModel()
            trainda = gT_model(toy_traindata)
            trainda_prediction = Variable(torch.argmax(trainda, 1))
            top = torch.sum(trainda_prediction == 0)
            down = torch.sum(trainda_prediction == 1)

        toy_trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(toy_traindata, trainda_prediction),
                                                      batch_size=args.batch_size, shuffle=True)
        valda = gT_model(toy_valdata)
        valda_prediction = torch.argmax(valda, 1)
        toy_valloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(toy_valdata, valda_prediction),
                                                    batch_size=args.val_num, shuffle=True)

        net.to(dvc)
        criterion.to(dvc)
        gT_model.to(dvc)
        train(toy_trainloader, net, opt, criterion, dvc)
        worse_fitness = test(toy_valloader, net, gT_model, worse_fitness, par, dvc)
        dict_fitness[par] = worse_fitness.view(1, 1).cpu()
    maxjob.release()

if __name__ == '__main__':

    if args.resume == 'True':
        pop = torch.load('./meta_checkpoint/parameteric_loss.pth')

    else:
        lst_parent_param = []
        for num_parent in range(NUM_PARENTS):
            meta_learner = MetaOptimizer()
            init_parameters = get_flat_parameters(meta_learner.rnn).reshape(1, -1)
            lst_parent_param.append(init_parameters)
        param_parent = torch.cat(lst_parent_param, 0)
        pop = dict(DNA=param_parent, MUT_STRENGTH=0.05 * torch.ones_like(param_parent))

    print(pop['DNA'])
    print(pop['MUT_STRENGTH'])

    record_var_ls1 = []
    record_acc_lst = []

    for outer_loop in range(args.outer_loop):
        pop = make_kids(pop, NUM_KIDS, pop['DNA'].shape[1])
        lst_thr = []
        for par in range(NUM_PARENTS + NUM_KIDS):
            meta_learner = MetaOptimizer(to_device=par % 2)
            set_parameters(meta_learner.rnn, pop['DNA'][par])
            meta_learner.eval()
            thr = Thread(target=ES, args=(meta_learner, par))
            thr.start()
            lst_thr.append(thr)
        for t in lst_thr:
            t.join()
        print(dict_fitness)
        print(dict_acc)
        lst_fitness = []
        lst_acc = []
        for nums in range(NUM_KIDS + NUM_PARENTS):
            lst_fitness.append(dict_fitness[nums])
            lst_acc.append(dict_acc[nums])

        fitness = torch.cat(lst_fitness, 1)
        acc = torch.cat(lst_acc, 1)

        idx = np.arange(pop['DNA'].shape[0])
        fitness = fitness.data.cpu().numpy()
        acc = acc.data.cpu().numpy()
        good_idx = idx[fitness.argsort()][0, -NUM_PARENTS:]
        for key in ['DNA', 'MUT_STRENGTH']:
            pop[key] = pop[key][good_idx]

        print(outer_loop, ': best fitness:', fitness[0][good_idx][-1], 'acc', acc[0][good_idx][-1])
        record_var_ls1.append(fitness[0][good_idx][-1])
        record_acc_lst.append(acc[0][good_idx][-1])

        print('saving figure...')
        if not os.path.isdir('train_imageFolder'):
            os.mkdir('train_imageFolder')
        if not os.path.isdir('meta_checkpoint'):
            os.mkdir('meta_checkpoint')

        plt.figure(dpi=200)
        plt.subplot(2, 1, 1)
        picture1, = plt.plot(np.arange(0, len(record_var_ls1)), record_var_ls1, color='red', linewidth=1.0,
                             linestyle='-')
        plt.legend(handles=[picture1], labels=['mse'], loc='best')
        plt.subplot(2, 1, 2)
        picture2, = plt.plot(np.arange(0, len(record_acc_lst)), record_acc_lst, color='blue', linewidth=1.0,
                             linestyle='-')
        plt.legend(handles=[picture2], labels=['val acc'], loc='best')
        plt.savefig('./train_imageFolder/parameteric_loss.jpg')
        plt.close()

        torch.save(pop, './meta_checkpoint/parameteric_loss.pth')
        print(
            '========================================================================================================================')