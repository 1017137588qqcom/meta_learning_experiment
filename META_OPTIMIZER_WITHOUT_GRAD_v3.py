import torch.nn.functional as F
import torch
import torch.nn as nn

class MSE_Derv(nn.Module):
    def __init__(self):
        super(MSE_Derv, self).__init__()
        self.kernel_size = 2
        self.transConv = 1
        self.padding = 0
        self.bias_flag = True

        self.functionlayers = nn.Sequential(
            nn.Linear(4, 32, bias=self.bias_flag),
            nn.PReLU(),
            nn.Linear(32, 64, bias=self.bias_flag),
            nn.PReLU(),
            nn.Linear(64, 128, bias=self.bias_flag),
            nn.PReLU(),
            nn.Linear(128, 256, bias=self.bias_flag),
            nn.PReLU(),
            nn.Linear(256, 512, bias=self.bias_flag),
            nn.PReLU(),
            nn.Linear(512, 1, bias=self.bias_flag),
            nn.Softplus()
        )
    def forward(self, x, y):
        t = torch.cat((x, y), 1)
        out = self.functionlayers(t)
        return out


class MetaOptimizer(nn.Module):

    def __init__(self, to_device='cpu', is_binary=True):
        super(MetaOptimizer, self).__init__()
        self.rnn = MSE_Derv()
        self.is_binary = is_binary
        self.to_device = to_device

    def forward(self, x, y, beta=1):
        targets_onehot = torch.zeros_like(x)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, y.long().unsqueeze(-1), 1).float()
        y_onehot = targets_onehot

        truth_pred = torch.sum(x * y_onehot, 1).view(-1, 1)
        false_pred = x * (1 - y_onehot)
        lst_all = []
        for i, sample in enumerate(false_pred):
            b = [t.view(1, 1) for i, t in enumerate(sample) if t != 0]
            c = torch.cat(b, 0).view(-1, 1)
            truth = torch.ones_like(c) * truth_pred[i][:]
            o0 = torch.cat((truth, c), 1)
            lst_all.append(o0)

        f_x = torch.cat(lst_all, 0)
        f_x = F.softmax(f_x)

        label = torch.zeros(f_x.shape[0], 2)
        label = (label + torch.tensor([1., 0.])).to(self.to_device)

        gru_input = f_x * 10  # .unsqueeze(1)
        y_label   = (label * 10).cuda(self.to_device)  # .unsqueeze(1)


        cost_tx = self.rnn(gru_input, y_label)

        return torch.mean(cost_tx)