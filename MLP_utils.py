import torch
import torch.nn as nn
import os
import pickle
import random


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1, activation_fn=nn.ReLU(), batchnorm=True):
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
    
def save_model(model, config, model_name, overwrite=False):
    dir_path = os.path.join('models', model_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif not overwrite:
        raise Exception(f"Model name already used. Call save_model with overwrite=True to overwrite.")
    
    torch.save(model.state_dict(), os.path.join(dir_path, 'state_dict.pth'))
    with open(os.path.join(dir_path, 'config.pkl'), "wb") as fp:
        pickle.dump(config, fp)    
    
def load_model(model_name, device):
    dir_path = os.path.join('models', model_name)

    if not os.path.exists(dir_path):
        raise Exception(f"No such model. Available models: {os.listdir('models')}.")
    
    with open(os.path.join(dir_path, 'config.pkl'), "rb") as fp:
        config = pickle.load(fp)

    hidden_sizes = config["depth"] * [config["width"]]
    model = SimpleMLP(input_size=config["input_size"], hidden_sizes=hidden_sizes, output_size=1, batchnorm=True).to(device)
    model.load_state_dict(torch.load(os.path.join(dir_path, 'state_dict.pth'), weights_only=True, map_location=device))
    model.eval()
    return model, config

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)