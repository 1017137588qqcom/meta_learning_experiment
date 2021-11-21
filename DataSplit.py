import torch
import random

class MultiToyTasks():
    def __init__(self, train_batchsize, val_batchsize):
        super(MultiToyTasks, self).__init__()
        self.train_traindataset = './data/multi_tasks/TrainData.pth'
        self.train_valdataset   = './data/multi_tasks/valData.pth'
        self.test_traindataset  = './data/multi_tasks/testTrainData.pth'
        self.test_valdataset    = './data/multi_tasks/testValData.pth'
        self.train_batchsize = train_batchsize
        self.val_batchsize   = val_batchsize

    def get_minibatchs_data(self, path, total_num):
        lst_val = torch.load(path)

        val_tuple = random.sample(range(0, lst_val.shape[0]), total_num)

        valX = lst_val[val_tuple, :]
        return valX

    def minibatch_train_sampler(self, num_batch):
        num_traindata = (num_batch * self.train_batchsize)
        num_valdata = self.val_batchsize
        train_minibatch = self.get_minibatchs_data(path=self.train_traindataset,
                                                          total_num=num_traindata)
        val_minibatch = self.get_minibatchs_data(path=self.train_valdataset,
                                                        total_num=num_valdata)

        return train_minibatch, val_minibatch

    def minibatch_test_sampler(self, num_batch):
        num_traindata = (num_batch * self.train_batchsize)
        num_valdata = self.val_batchsize
        train_minibatch_loader = self.get_minibatchs_data(path=self.test_traindataset,
                                                          total_num=num_traindata)
        val_minibatch_loader = self.get_minibatchs_data(path=self.test_valdataset,
                                                        total_num=num_valdata)

        return train_minibatch_loader, val_minibatch_loader