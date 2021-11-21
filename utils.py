# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from functools import reduce
from operator import mul
import random
import torch.nn.functional as F
import csv
import os
import sys
import time
import math

term_width = 100
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def set_parameters(meta_networks, grad_list, bias=True):
    offset = 0
    if bias == True:
        for mod in meta_networks.children():
            '''拿到第一个children，判断一下这个children是不是nn.Sequential'''
            if isinstance(mod, nn.Sequential):
                for m in mod.children():
                    if len(m._parameters) is not 0:
                        if isinstance(m, nn.PReLU):
                            weight_shape = m._parameters['weight'].size()
                            weight_flat_size = reduce(mul, weight_shape, 1)
                            m._parameters['weight'].data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
                            offset += weight_flat_size
                        else:
                            weight_shape = m._parameters['weight'].size()
                            bias_shape = m._parameters['bias'].size()

                            weight_flat_size = reduce(mul, weight_shape, 1)
                            bias_flat_size = reduce(mul, bias_shape, 1)

                            m._parameters['weight'].data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
                            m._parameters['bias'].data = grad_list[offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)
                            offset += weight_flat_size + bias_flat_size
            else:
                if len(mod._parameters) is not 0:
                    if isinstance(mod, nn.PReLU):
                        weight_shape = mod._parameters['weight'].size()
                        weight_flat_size = reduce(mul, weight_shape, 1)
                        mod._parameters['weight'].data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
                        offset += weight_flat_size
                    else:
                        weight_shape = mod._parameters['weight'].size()
                        bias_shape = mod._parameters['bias'].size()

                        weight_flat_size = reduce(mul, weight_shape, 1)
                        bias_flat_size = reduce(mul, bias_shape, 1)
                        mod._parameters['weight'].data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
                        mod._parameters['bias'].data = grad_list[offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)
                        offset += weight_flat_size + bias_flat_size
    else:
        for mod in meta_networks.children():
            '''拿到第一个children，判断一下这个children是不是nn.Sequential'''
            if isinstance(mod, nn.Sequential):
                for m in mod.children():
                    if len(m._parameters) is not 0:
                        if isinstance(m, nn.PReLU):
                            weight_shape = m._parameters['weight'].size()
                            weight_flat_size = reduce(mul, weight_shape, 1)
                            m._parameters['weight'].data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
                            offset += weight_flat_size
                        else:
                            weight_shape = m._parameters['weight'].size()

                            weight_flat_size = reduce(mul, weight_shape, 1)

                            m._parameters['weight'].data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
                            offset += weight_flat_size
            else:
                if len(mod._parameters) is not 0:
                    if isinstance(mod, nn.PReLU):
                        weight_shape = mod._parameters['weight'].size()
                        weight_flat_size = reduce(mul, weight_shape, 1)
                        mod._parameters['weight'].data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
                        offset += weight_flat_size
                    else:
                        weight_shape = mod._parameters['weight'].size()

                        weight_flat_size = reduce(mul, weight_shape, 1)
                        mod._parameters['weight'].data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
                        offset += weight_flat_size

def get_flat_parameters(meta_opt, bias=True):
    '''
    获取元学习器的梯度信息
    :param meta_opt: 元学习器的模型
    :return: 展开的梯度信息
    '''
    # print(meta_opt.children)
    _loss_grad = []
    for mod in meta_opt.children():  # model_with_grad.module.children()
        # print(isinstance(mod, nn.Sequential))
        '''拿到第一个children，判断一下这个children是不是nn.Sequential'''
        if isinstance(mod, nn.Sequential):
            for m in mod.children():
                # print(m)
                if len(m._parameters) is not 0:
                    if isinstance(m, nn.PReLU):
                        _loss_grad.append(m._parameters['weight'].data.view(-1).unsqueeze(-1))
                    else:
                        _loss_grad.append(m._parameters['weight'].data.view(-1).unsqueeze(-1))
                        if bias:
                            _loss_grad.append(m._parameters['bias'].data.view(-1).unsqueeze(-1))
        else:
            # print(mod)
            if len(mod._parameters) is not 0:
                if isinstance(mod, nn.PReLU):
                    _loss_grad.append(mod._parameters['weight'].data.view(-1).unsqueeze(-1))
                else:
                    _loss_grad.append(mod._parameters['weight'].data.view(-1).unsqueeze(-1))
                    if bias:
                        _loss_grad.append(mod._parameters['bias'].data.view(-1).unsqueeze(-1))

    flat_loss_grad = torch.cat(_loss_grad).squeeze(-1).unsqueeze(0).unsqueeze(1)
    # print(flat_loss_grad)
    return flat_loss_grad

class CSV_writer():
    def __init__(self, path):
        self.file = open(path,'w',encoding='utf-8')
        self.writer=csv.writer(self.file)
    def Writer(self, content):
        self.writer.writerow([str(content.data.cpu().numpy())])
    def close(self):
        self.file.close()

def reset_parameters(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.uniform_(m.weight, a=-5, b=5)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-5, b=5)

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
