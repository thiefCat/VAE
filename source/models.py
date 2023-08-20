import torch
import torch.nn as nn
from source import utils
import math



class FCEncoder(nn.Module):
    '''
    (batch, 28, 28) --> (batch, 2 x z_dim)
    hidden_sizes: list
    z_dim: dimension of the z space
    '''
    def __init__(self, z_dim, input_size=784, hidden_sizes=[500, 200, 50], activation=nn.ELU()):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.z_dim = z_dim
        self.encode_layers = self.build_layers()

    def build_layers(self):
        layers = []
        prev_size = self.input_size
        for layer_id, size in enumerate(self.hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            layers.append(self.activation)
            prev_size = size
        layers.append(nn.Linear(prev_size, 2 * self.z_dim))
        return nn.Sequential(*layers)
    
    def encode(self, input):
        '''
        input: [batch, 28, 28]
        return: ([batch, z_dim], [batch, z_dim])
        '''
        input = input.view(input.shape[0], -1)
        mv = self.encode_layers(input)  # [batch, 2*z_dim]
        m, v = utils.gaussian_parameters(mv, -1)
        return m, v
    
    def forward(self, input):
        return self.encode(input)


class FCDecoder(nn.Module):
    '''
    (batch, z_dim) --> (batch, 28, 28)
    hidden_sizes: list
    z_dim: dimension of the z space
    '''
    def __init__(self, z_dim, out_size=784, hidden_sizes=[50, 200, 500], activation=nn.ELU()):
        super().__init__()
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.z_dim = z_dim
        self.decode_layers = self.build_layers()

    def build_layers(self):
        layers = []
        prev_size = self.z_dim
        for layer_id, size in enumerate(self.hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            layers.append(self.activation)
            prev_size = size
        layers.append(nn.Linear(prev_size, self.out_size))
        return nn.Sequential(*layers)
    
    def decode(self, input):
        '''
        input: [batch, z_dim]
        return: [batch, 28, 28]
        '''
        out = self.decode_layers(input)
        out = torch.sigmoid(out)  # make output between to fit in [0, 1]
        dim = int(math.sqrt(out.shape[-1]))
        return out.view(out.shape[0], dim, dim)
    
    def forward(self, input):
        return self.decode(input)




class CNNEncoder(nn.Module):
    '''
    (batch, 28, 28) --> (batch, 2 x z_dim)
    z_dim: dimension of the z space
    BatchNorm2d: [b, num_filter, h, w] --> mean: [1, num_filter, 1, 1]
    '''
    def __init__(self, z_dim=2, activation=nn.ELU()):
        super().__init__()
        self.activation = activation
        self.z_dim = z_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1), # (1, 28, 28) --> (32, 14, 14)
            nn.BatchNorm2d(32),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # --> (64, 7, 7)
            nn.BatchNorm2d(64),
            activation,
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),  # --> (32, 4, 4)
            nn.BatchNorm2d(32),
            activation,
        )
        self.fc = nn.Linear(32*4*4, 2*z_dim)
    
    def encode(self, input):
        '''
        input: [batch, 28, 28]
        return: ([batch, z_dim], [batch, z_dim])
        '''
        input = input.unsqueeze(1) # [batch, 28, 28] --> [batch, 1, 28, 28]
        batch_size = input.shape[0]
        feature_maps = self.conv_layers(input) 
        flattened = feature_maps.view(batch_size, -1)  # [batch, 8*3*3]
        mv = self.fc(flattened) # [batch, 2*z_dim]
        m, v = utils.gaussian_parameters(mv, -1)
        return m, v
    
    def forward(self, input):
        return self.encode(input)


class CNNDecoder(nn.Module):
    '''
    (batch, z_dim) --> (batch, 28, 28)
    z_dim: dimension of the z space
    '''
    def __init__(self, z_dim, activation=nn.ELU()):
        super().__init__()
        self.activation = activation
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 32*4*4)
        self.conv_transpose_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # (32, 4, 4) --> (64, 7, 7)
            nn.BatchNorm2d(64),
            activation,
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),  # --> (32, 14, 14)
            nn.BatchNorm2d(32),
            activation,
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),  # --> (32, 28, 28)
            nn.BatchNorm2d(32),
            activation,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1) # --> (1, 28, 28)
        )
    
    def decode(self, input):
        '''
        input: [batch, z_dim]
        return: [batch, 28, 28]
        '''
        out = self.fc(input)
        out = out.view(out.shape[0], 32, 4, 4)
        out = self.conv_transpose_layers(out)
        out = out.squeeze(1)  # (batch, 1, 28, 28) --> (batch, 28, 28)
        out = torch.sigmoid(out)  # make output between to fit in [0, 1]
        return out
    
    def forward(self, input):
        return self.decode(input)



class VAE(nn.Module):
    # input: 28x28
    # c: p(x|z) = N(decoder(z), cI)
    def __init__(self, c=0.05, z_dim=2, v='fc'):
        super().__init__()
        self.z_dim = z_dim
        if v == 'fc':
            self.enc = FCEncoder(self.z_dim)
            self.dec = FCDecoder(self.z_dim)
        elif v == 'cnn':
            self.enc = CNNEncoder(self.z_dim)
            self.dec = CNNDecoder(self.z_dim)
        self.c = c

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    
    def negative_elbo(self, input):
        '''
        input: [batch, 28, 28]
        '''
        m, v = self.enc.encode(input)
        z = utils.sample_gaussian(m, v)    # (batch, z_dim)
        kl = utils.kl_normal(m, v, self.z_prior_m, self.z_prior_v)   # (batch,)
        out = self.dec.decode(z)  # (batch, 28, 28)
        rec = -torch.norm(input - out, p=2, dim=[1, 2])**2 / (2 * self.c) # (batch,)
        kl = kl.mean()
        rec = rec.mean()
        loss = -rec + kl
        return loss, -rec, kl
    
    def sample_given_z(self, z):
        out = self.dec.decode(z)
        return out

    def sample(self, batch):
        z = self.sample_z(batch)
        m = self.sample_given_z(z)
        # return torch.normal(m, math.sqrt(self.c))
        return m

    def sample_z(self, batch):
        return utils.sample_gaussian(
            self.z_prior_m.expand(batch, self.z_dim),
            self.z_prior_v.expand(batch, self.z_dim))
    
    def generate_z_grid(self, grid_size):
        x = torch.linspace(-2, 2, steps=grid_size)
        y = torch.linspace(2, -2, steps=grid_size)  

        xv, yv = torch.meshgrid(x, y, indexing='xy')
        # print(xv)
        # print(yv)
        # Reshape the grid to have shape (400, 2)
        grid = torch.stack((xv.flatten(), yv.flatten()), dim=1)
        return grid