import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn=nn.ReLU(), batchnorm=False):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential()
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(activation_fn)
            input_size = hidden_size
        self.layers.append(nn.Linear(input_size, output_size))
    
    def forward(self, x):
        out = self.layers(x)
        return out
    

class LNLoss(nn.Module):
    def __init__(self, N):
        super(LNLoss, self).__init__()
        self.N = N
    
    def forward(self, output, target):
        loss = torch.mean(torch.abs(output - target)**self.N)
        return loss