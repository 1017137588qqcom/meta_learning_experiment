import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from utils import progress_bar
from model import LeNet_FashionMNIST, ResNet18, LeNet_Cifar10, VGG16
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_epoch', default=200, type=int, help='max epoch')
parser.add_argument('--flod', '-f', default=1, type=int, help='test flod')
parser.add_argument('--Network', '-r', default='VGG16', help='classifier:LeNet-Mnist, ResNet18, VGG16, LeNet_Cifar10')
parser.add_argument('--dataset', default='Cifar10', help='dataset: Mnist, Cifar10')
args = parser.parse_args()

gpu = "0"           # which GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr_loss = 1000
best_acc = 0        # best test accuracy
start_epoch = 0     # start from epoch 0 or last checkpoint epoch
flod = args.flod
work = True
weight_decay = 0.000

def data_prepare():
    if args.dataset == 'Cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    elif args.dataset == 'Mnist':
        trainset = torchvision.datasets.MNIST('./data', download=True, train=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))]))
        testset = torchvision.datasets.MNIST('./data', download=True, train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader

def model_prepare(work):
    print('==> Building model..')
    global best_acc
    global start_epoch
    if work == True:
        if args.Network == 'LeNet-Mnist':
            net = LeNet_FashionMNIST()
        elif args.Network == 'ResNet18':
            net = ResNet18()
        elif args.Network == 'LeNet_Cifar10':
            net = LeNet_Cifar10()
        elif args.Network == 'VGG16':
            net = VGG16()
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4, threshold_mode='rel')

    criterion = nn.CrossEntropyLoss()
    return net, optimizer, scheduler, criterion

def train(epoch, dataloader, net, optimizer, criterion, vali=True):
    """Train the network"""
    print('\nEpoch: %d' % epoch)
    global tr_loss
    net.train()
    num_id = 0
    train_loss = 0
    correct = 0
    total = 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # if batch_id < (12800 / args.batch_size):
        num_id += 1
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets) # .long()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                     % (train_loss / (batch_id + 1), 100. * correct / total, correct, total))

    if vali is True:
        tr_loss = train_loss / num_id

    return train_loss / num_id, 100. * correct / total

def test(epoch, dataloader, net, criterion, vali=True):
    """Validation and the test."""
    global best_acc
    net.eval()
    num_id = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(dataloader):
            # if batch_id < (2560 / args.batch_size):
            num_id += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()

            _, predicted = outputs.max(1)  # judge max elements in predicted`s Row(1:Row     0:Column)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  # judge how many elements same in predicted and targets

            progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_id + 1), 100. * correct / total, correct, total))
    if vali is True:
        # Save checkpoint.
        acc = 100. * correct / total
        if acc >= best_acc:
            print('Saving:')
            state1 = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('best_acc'):
                os.mkdir('best_acc')
            torch.save(state1, './best_acc/best_CE''.t7')
            best_acc = acc
            best_acc1 = open('./test_dir/CE.txt', 'w')
            best_acc1.write(str(best_acc))
            best_acc1.close()
    return test_loss / num_id, 100. * correct / total

if __name__ == '__main__':
    for avg in range(10):
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        trainloader, testloader = data_prepare()
        net, optimizer, scheduler, criterion = model_prepare(work)
        train_list0, train_list1 = [], []
        test_list0, test_list1 = [], []
        for epoch in range(start_epoch, start_epoch+args.max_epoch):
            train_loss, train_acc = train(epoch, trainloader, net, optimizer, criterion)
            test_loss, test_acc = test(epoch, testloader, net, criterion)

            train_list0.append(train_loss)
            train_list1.append(train_acc)
            test_list0.append(test_loss)
            test_list1.append(test_acc)

            scheduler.step(tr_loss)
            lr = optimizer.param_groups[0]['lr']
            print('Saving:')
            state3 = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('finial'):
                os.mkdir('finial')
            torch.save(state3, './finial/CE'+'.t7')

            ### 这个要存成csv
            train_array0 = np.array(train_list0)
            train_array1 = np.array(train_list1)
            test_array0 = np.array(test_list0)
            test_array1 = np.array(test_list1)


            plt.figure(figsize=(10,10), dpi=180)
            plt.subplot(2, 2, 1)
            plt.xlabel('step')
            plt.ylabel('train loss')
            plt.plot([i for i in range(epoch+1)], train_array0, '-')
            plt.subplot(2, 2, 2)
            plt.xlabel('step')
            plt.ylabel('train acc')
            plt.plot([i for i in range(epoch+1)], train_array1, '-')
            plt.subplot(2, 2, 3)
            plt.xlabel('step')
            plt.ylabel('test loss')
            plt.plot([i for i in range(epoch+1)], test_array0, '-')
            plt.subplot(2, 2, 4)
            plt.xlabel('step')
            plt.ylabel('test acc')
            plt.plot([i for i in range(epoch+1)], test_array1, '-')
            plt.tight_layout()
            plt.savefig('./test_dir/CE'+str(avg)+'.jpg')
            plt.close()

        train_array0 = np.array(train_list0)
        train_array1 = np.array(train_list1)
        test_array0 = np.array(test_list0)
        test_array1 = np.array(test_list1)
        lst_store = [train_array0, train_array1, test_array0, test_array1]
        file = pd.DataFrame(data=lst_store)
        file.to_csv('./test_dir/CE'+str(avg)+'.csv')