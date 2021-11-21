import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from META_OPTIMIZER_WITHOUT_GRAD_v3 import MetaOptimizer
from model import ToyModel, uniform_ToyModel
from utils import set_parameters
from DataSplit import MultiToyTasks
import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=500, type=int, help='batch size')
parser.add_argument('--val_num', default=10000, type=int, help='batch size')
parser.add_argument('--max_epoch', default=10, type=int, help='max epoch')
parser.add_argument('--num_task', default=1, type=int, help='amount of task to obtain average')
parser.add_argument('--inner_loop', default=100, type=int, help='inner loop')
parser.add_argument('--flod', '-f', default=1, type=int, help='test flod')
parser.add_argument('--Network', '-r', default='single_layer', help='classifier:single_layer, three_layer')
args = parser.parse_args()

gpu = "0"           # which GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
to_dvice = 'cuda' if torch.cuda.is_available() else 'cpu'
tr_loss = 1000
flod = args.flod
work = True
weight_decay = 0.000

def init_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.cudnn_enabled = False

init_seed(seed=1)

datas = MultiToyTasks(train_batchsize=args.batch_size, val_batchsize=args.val_num)

lst_ce_test    = []
lst_mse_test   = []
lst_meta_test  = []

lst_meta_loss_ce = []
lst_meta_loss_mse = []
lst_meta_loss_mln = []

lst_ce_train_loss   = []
lst_ce_test_loss    = []
lst_mse_train_loss  = []
lst_mse_test_loss   = []
lst_meta_train_loss = []
lst_meta_test_loss  = []

pop = torch.load('./meta_checkpoint/parameteric_loss.pth')
print(pop['MUT_STRENGTH'])
print(pop['DNA'])

meta_learner = MetaOptimizer(to_device=to_dvice, is_binary=True)
set_parameters(meta_learner.rnn, pop['DNA'][24], bias=True)
meta_learner = meta_learner.cuda(to_dvice)
meta_learner.eval()
print(meta_learner.rnn)

for tttt in range(args.max_epoch * args.inner_loop):
    lst_ce_test.append(torch.zeros((1,1)))
    lst_mse_test.append(torch.zeros((1,1)))
    lst_meta_test.append(torch.zeros((1,1)))

    lst_ce_train_loss.append(torch.zeros((1, 1)))
    lst_ce_test_loss.append(torch.zeros((1, 1)))
    lst_mse_train_loss.append(torch.zeros((1, 1)))
    lst_mse_test_loss.append(torch.zeros((1, 1)))
    lst_meta_train_loss.append(torch.zeros((1, 1)))
    lst_meta_test_loss.append(torch.zeros((1, 1)))

    lst_meta_loss_ce.append(torch.zeros((1, 1)))
    lst_meta_loss_mse.append(torch.zeros((1, 1)))
    lst_meta_loss_mln.append(torch.zeros((1, 1)))

for tttt in range(args.num_task):
    top = 50000
    down = 0
    while abs(top - down) >= 5000:
        groundTruth = ToyModel().cuda(to_dvice)
        toy_traindata, toy_valdata = datas.minibatch_test_sampler(num_batch=args.inner_loop)
        trainda = groundTruth(toy_traindata.cuda(to_dvice))
        trainda_prediction = Variable(torch.argmax(trainda, 1))
        top = torch.sum(trainda_prediction == 0)
        down = torch.sum(trainda_prediction == 1)
        print(top, down)

    torch.save(groundTruth, './meta_checkpoint/gtm.pth')

    valda = groundTruth(toy_valdata.cuda(to_dvice))
    valda_prediction = torch.argmax(valda, 1)
    toy_trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(toy_traindata, trainda_prediction),
                                                  batch_size=args.batch_size, shuffle=True)
    toy_valloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(toy_valdata, valda_prediction),
                                                batch_size=args.val_num, shuffle=True)

    m = ToyModel()
    m = m.cuda(to_dvice)
    opt_ce = optim.Adam(m.parameters(), lr=1e-3)
    ls = nn.CrossEntropyLoss()
    for num_task in range(args.max_epoch):
        for idx, (x, y) in enumerate(toy_trainloader):
            x = x.cuda(to_dvice)
            y = y.cuda(to_dvice)
            f_x = m(x)

            l = ls(f_x, y.long())

            opt_ce.zero_grad()
            l.backward()
            opt_ce.step()

            prediction_test = 0
            val_ls = 0
            ce_ls = 0
            for ii, (vx, vy) in enumerate(toy_valloader):
                vx = vx.cuda(to_dvice)
                vy = vy.cuda(to_dvice)
                update_f_tx = m(vx).data
                ce_ls += ls(update_f_tx, vy.long()).data/len(toy_valloader)
                gt_f_tx = groundTruth(vx).data
                val_ls += (F.mse_loss(F.softmax(update_f_tx), F.softmax(gt_f_tx))).data/len(toy_valloader)
                prediction = torch.argmax(update_f_tx, 1)
                prediction_test += ((prediction == vy.long()).sum().float() / vy.shape[0]) / len(toy_valloader)

            lst_ce_test[num_task*args.inner_loop+idx] += (prediction_test.data / args.num_task)
            lst_ce_train_loss[num_task*args.inner_loop+idx] += (l.data / args.num_task)
            lst_ce_test_loss[num_task*args.inner_loop+idx] += (ce_ls.data / args.num_task)
            lst_meta_loss_ce[num_task*args.inner_loop+idx] += (val_ls.data / args.num_task)
    print('CE:', num_task, idx, ':', prediction_test, val_ls)
    print('---------------------------------------------------------')
    torch.save(m, './meta_checkpoint/m_CE.pth')
    m = ToyModel()
    m = m.cuda(to_dvice)
    opt_mse = optim.Adam(m.parameters(), lr=1e-3)
    for num_task in range(args.max_epoch):
        for idx, (x, y) in enumerate(toy_trainloader):
            x = x.cuda(to_dvice)
            y = y.cuda(to_dvice)
            f_x = m(x)

            targets_onehot = torch.zeros_like(f_x)
            targets_onehot.zero_()
            targets_onehot.scatter_(1, y.long().unsqueeze(-1), 1).float()
            y_onehot = targets_onehot

            ls = F.mse_loss(F.softmax(f_x), y_onehot)

            opt_mse.zero_grad()
            ls.backward()
            opt_mse.step()

            prediction_test = 0
            val_ls = 0
            mse_ls = 0
            for ii, (vx, vy) in enumerate(toy_valloader):
                vx = vx.cuda(to_dvice)
                vy = vy.cuda(to_dvice)
                update_f_tx = m(vx).data
                targets_onehot = torch.zeros_like(update_f_tx)
                targets_onehot.zero_()
                targets_onehot.scatter_(1, vy.long().unsqueeze(-1), 1).float()
                y_onehot = targets_onehot
                mse_ls += F.mse_loss(F.softmax(update_f_tx), y_onehot).data/len(toy_valloader)

                gt_f_tx = groundTruth(vx).data
                val_ls += (F.mse_loss(F.softmax(update_f_tx), F.softmax(gt_f_tx))).data/len(toy_valloader)
                prediction = torch.argmax(update_f_tx, 1)
                prediction_test += ((prediction == vy.long()).sum().float() / vy.shape[0]) / len(toy_valloader)

            lst_mse_test[num_task * args.inner_loop + idx] += (prediction_test.data / args.num_task)
            lst_mse_train_loss[num_task * args.inner_loop + idx] += (ls.data / args.num_task)
            lst_mse_test_loss[num_task * args.inner_loop + idx] += (mse_ls.data / args.num_task)
            lst_meta_loss_mse[num_task * args.inner_loop + idx] += (val_ls.data / args.num_task)
    print('MSE:', num_task, idx, ':', prediction_test)
    print('---------------------------------------------------------')
    torch.save(m, './meta_checkpoint/m_MSE.pth')
    m = ToyModel()
    m = m.cuda(to_dvice)
    opt_mln = optim.Adam(m.parameters(), lr=1e-3)
    for num_task in range(args.max_epoch):
        for idx, (x, y) in enumerate(toy_trainloader):
            x = x.cuda(to_dvice)
            y = y.cuda(to_dvice)
            f_x = m(x)

            cost = meta_learner(x=f_x, y=y)
            cost_loss = torch.mean(cost)
            opt_mln.zero_grad()
            cost_loss.backward()
            opt_mln.step()

            prediction_test = 0
            mln_ls = 0
            val_ls = 0
            for ii, (vx, vy) in enumerate(toy_valloader):
                vx = vx.cuda(to_dvice)
                vy = vy.cuda(to_dvice)
                update_f_tx = m(vx).data
                mln_ls += meta_learner(x=update_f_tx, y=vy).data / len(toy_valloader)

                prediction = torch.argmax(update_f_tx, 1)
                prediction_test += ((prediction == vy.long()).sum().float() / vy.shape[0]) / len(toy_valloader)

                gt_f_tx = groundTruth(vx).data
                val_ls += (F.mse_loss(F.softmax(update_f_tx), F.softmax(gt_f_tx))).data/len(toy_valloader)

            lst_meta_test[num_task * args.inner_loop + idx] += (prediction_test.data / args.num_task)
            lst_meta_train_loss[num_task * args.inner_loop + idx] += (cost_loss.data / args.num_task)
            lst_meta_test_loss[num_task * args.inner_loop + idx] += (mln_ls.data / args.num_task)
            lst_meta_loss_mln[num_task * args.inner_loop + idx] += (val_ls.data / args.num_task)
            print('MLN:', num_task, idx, 'prediction_test:', prediction_test) #, 'val mse', mse.data.cpu().numpy(), 'train mse', mse.data.cpu().numpy()  cost_loss,
            print('---------------------------------------------------------')
    torch.save(m, './meta_checkpoint/m_MLN.pth')
    print('==============================================================')

lst = []
for d in lst_ce_test:
    lst.append(d.view(1,1).cpu())
test_acc_ce = torch.cat(lst)

lst = []
for d in lst_mse_test:
    lst.append(d.view(1,1).cpu())
test_acc_mse = torch.cat(lst)

lst = []
for d in lst_meta_test:
    lst.append(d.view(1,1).cpu())
test_acc_mln = torch.cat(lst)

lst = []
for d in lst_meta_loss_ce:
    lst.append(d.view(1,1).cpu())
meta_loss_ce = torch.cat(lst)

lst = []
for d in lst_meta_loss_mse:
    lst.append(d.view(1,1).cpu())
meta_loss_mse = torch.cat(lst)

lst = []
for d in lst_meta_loss_mln:
    lst.append(d.view(1,1).cpu())
meta_loss_mln = torch.cat(lst)
###############
lst = []
for d in lst_ce_train_loss:
    lst.append(d.view(1,1).cpu())
ce_train_loss = torch.cat(lst)

lst = []
for d in lst_ce_test_loss:
    lst.append(d.view(1,1).cpu())
ce_test_loss = torch.cat(lst)

lst = []
for d in lst_mse_train_loss:
    lst.append(d.view(1,1).cpu())
mse_train_loss = torch.cat(lst)

lst = []
for d in lst_mse_test_loss:
    lst.append(d.view(1,1).cpu())
mse_test_loss = torch.cat(lst)

lst = []
for d in lst_meta_train_loss:
    lst.append(d.view(1,1).cpu())
meta_train_loss = torch.cat(lst)

lst = []
for d in lst_meta_test_loss:
    lst.append(d.view(1,1).cpu())
meta_test_loss = torch.cat(lst)

c = torch.cat((test_acc_ce, test_acc_mse), 1)
c = torch.cat((c, test_acc_mln), 1)
c = torch.cat((c, meta_loss_ce), 1)
c = torch.cat((c, meta_loss_mse), 1)
c = torch.cat((c, meta_loss_mln), 1)

c = torch.cat((c, ce_train_loss), 1)
c = torch.cat((c, ce_test_loss), 1)
c = torch.cat((c, mse_train_loss), 1)
c = torch.cat((c, mse_test_loss), 1)
c = torch.cat((c, meta_train_loss), 1)
c = torch.cat((c, meta_test_loss), 1)

a = c.detach().numpy()
np.savetxt('./test_dir/MetaTestingResult_generated.csv',a,delimiter=',')

plt.figure(figsize=(10,10), dpi=180)

p1, =plt.plot(np.arange(0, len(lst_meta_test)), lst_meta_test, color='xkcd:red', linewidth=2.0,
                     linestyle='-', alpha=1)
p2, =plt.plot(np.arange(0, len(lst_mse_test)), lst_mse_test, color='xkcd:green', linewidth=2.0,
                     linestyle='-', alpha=1)
p3, =plt.plot(np.arange(0, len(lst_ce_test)), lst_ce_test, color='xkcd:blue', linewidth=2.0,
                     linestyle='-', alpha=1)
plt.legend([p1, p2, p3], ["MLN", "MSE", "CE"], loc='best')

plt.tight_layout()
plt.savefig('./test_dir/MetaTestingResult_generated_acc.jpg')
plt.close()

plt.figure(figsize=(10,10), dpi=180)
f, ax = plt.subplots(2, 2)
ax[0][0].set_title('CE loss')
p1, =ax[0][0].plot(np.arange(0, len(lst_ce_train_loss)), lst_ce_train_loss, color='xkcd:pink', linewidth=1.0,
                     linestyle='-', alpha=0.5)
p2, =ax[0][0].plot(np.arange(0, len(lst_ce_test_loss)), lst_ce_test_loss, color='xkcd:red', linewidth=2.0,
                     linestyle='-', alpha=1)
plt.legend([p1, p2], ["train loss", "test loss"], loc='best')

ax[0][1].set_title('meta loss')
p1, =ax[0][1].plot(np.arange(0, len(lst_mse_train_loss)), lst_mse_train_loss, color='xkcd:light blue', linewidth=1.0,
                     linestyle='-', alpha=0.5)
p2, =ax[0][1].plot(np.arange(0, len(lst_mse_test_loss)), lst_mse_test_loss, color='xkcd:blue', linewidth=2.0,
                     linestyle='-', alpha=1)
plt.legend([p1, p2], ["train loss", "test loss"], loc='best')

ax[1][0].set_title('mse loss')
p1, =ax[1][0].plot(np.arange(0, len(lst_meta_train_loss)), lst_meta_train_loss, color='xkcd:orange', linewidth=1.0,
                     linestyle='-', alpha=0.5)
p2, =ax[1][0].plot(np.arange(0, len(lst_meta_test_loss)), lst_meta_test_loss, color='xkcd:orange', linewidth=2.0,
                     linestyle='-', alpha=1)
plt.legend([p1, p2], ["train loss", "test loss"], loc='best')


ax[1][1].set_title('Compare the meta-loss of MLN, CE, and MSE')
p1, =ax[1][1].plot(np.arange(0, len(lst_meta_loss_ce)), lst_meta_loss_ce, color='xkcd:red', linewidth=2.0,
                     linestyle='-', alpha=1)
p2, =ax[1][1].plot(np.arange(0, len(lst_meta_loss_mse)), lst_meta_loss_mse, color='xkcd:green', linewidth=2.0,
                     linestyle='-', alpha=1)
p3, =ax[1][1].plot(np.arange(0, len(lst_meta_loss_mln)), lst_meta_loss_mln, color='xkcd:blue', linewidth=2.0,
                     linestyle='-', alpha=1)
plt.legend([p1, p2, p3], ["CE", "MSE", "MLN"], loc='best')

plt.tight_layout()
plt.savefig('./test_dir/MetaTestingResult_generated_loss_metaloss.jpg')
plt.close()