import numpy as np
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, nc=1, num_layers=5, nh=128, nz=200, device = "cuda:0"):
        super(VAE, self).__init__()

        self.nz = nz
        self.device = device

        ##---------##
        ## ENCODER ##
        ##---------##

        self.encoder = nn.Sequential()

        # add conv layer blocks: each block halves image size and doubles channels
        for i in range(num_layers):
            in_channels = nc if i==0 else nh*2**(i-1)
            out_channels = nh*2**i
            for layer in self.convblock(in_channels, out_channels):
                self.encoder.append(layer)

        # Flatten before passing to linear layers
        self.encoder.append(nn.Flatten())
        self.encoder.append(nn.Linear(nh*2**(num_layers-1), nh*2**(num_layers-1)))

        ##---------##
        ##   MU    ##
        ##---------##

        self.mu = nn.Sequential(
            nn.Linear(nh*2**(num_layers-1), nz),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        ##---------##
        ## LOGVAR  ##
        ##---------##

        self.log_var = nn.Sequential(
            nn.Linear(nh*2**(num_layers-1), nz),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        ##---------##
        ## DECODER ##
        ##---------##

        self.decoder = nn.Sequential(
            nn.Linear(nz, nh*2**(num_layers-1)),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Unflatten(1, (nh*2**(num_layers-1), 1, 1)))

        # add conv transpose layer blocks: each block doubles image size and halves channels
        for i in range(num_layers-1):
            in_channels = nh*2**(num_layers-i-1)
            out_channels = nh*2**(num_layers-i-2)
            for layer in self.transconvblock(in_channels, out_channels):
                self.decoder.append(layer)

        # last layer is different
        self.decoder.append(nn.ConvTranspose2d(nh, nc, 4, 2, 1, bias=False))

    def transconvblock(self, in_channels, out_channels):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.LeakyReLU(True)]
        return layers

    def convblock(self, in_channels, out_channels):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.LeakyReLU(True)]
        return layers

    def parameterize(self, mu, log_var):
        epsilon = torch.Tensor(np.random.normal(size=(self.nz), scale=1.0)).to(self.device)
        return mu + epsilon * torch.exp(log_var / 2)

    def forward(self, input):
        x = self.encoder(input)
        mu_x = self.mu(x)
        log_var_x = self.log_var(x)
        z = self.parameterize(mu_x, log_var_x)
        output = self.decoder(z)
        output = output.view(*input.shape)
        return mu_x, log_var_x, output

    def sample(self, batch_size):

        z = torch.randn(batch_size, self.nz)
        z = z.to(self.device)
        samples = self.decoder(z)

        return samples