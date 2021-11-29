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

        x = F.softmax(x)
        gru_input = x * 10
        y_label   = (y_onehot * 10).cuda(self.to_device)

        cost_tx = self.rnn(gru_input, y_label)

        return torch.mean(cost_tx)