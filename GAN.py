import numpy as np
import torch
import torch.nn as nn

## Generator network class ##

class Generator(nn.Module):
    def __init__(self, input_size=32, nc=1, ngf=128, nz=200):
        super(Generator, self).__init__()

        """  Initializes a generator network for a GAN with the given parameters, 
             with the number of layers determined by input size

        Args:
            input_size (int): Size of the 2D input image (assumed to be square).
            nc (int)        : Number of channels in the input image. 
            ngf (int)       : Number of 'features' or channels used by the generator network in last layer.
            nz (int)        : Size of the latent vector used to generate an image.

        Returns:
            None

        """

        self.input_size = input_size
        self.nc = nc
        self.ngf = ngf
        self.num_layers = int(np.log(self.input_size)/np.log(2))
        self.nz = nz

        self.layers = nn.Sequential(
            nn.Linear(nz, ngf*2**(self.num_layers-1)),
            nn.LeakyReLU(True),
            nn.Unflatten(1, (ngf*2**(self.num_layers-1), 1, 1))
        )

        # add conv transpose layer blocks: each block doubles image size and halves channels
        for i in range(self.num_layers-1):
            in_channels = ngf*2**(self.num_layers-i-1)
            out_channels = ngf*2**(self.num_layers-i-2)
            for layer in self.transconvblock(in_channels, out_channels):
                self.layers.append(layer)

        # last layer block is different
        self.layers.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1))

    def transconvblock(self, in_channels, out_channels):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                  nn.BatchNorm2d(out_channels),
                  nn.LeakyReLU(True)]
        return layers

    def forward(self, input):
        output = self.layers(input)
        return output
    
    
## Discriminator network class ##

class Discriminator(nn.Module):
    def __init__(self, input_size=32, nc=1, ndf=128):
        super(Discriminator, self).__init__()

        """  Initializes a discriminator network for a GAN with the given parameters, 
             with the number of layers determined by input size.

        Args:
            input_size (int): Size of the 2D input image (assumed to be square).
            nc (int)        : Number of channels in the input image. 
            ndf (int)       : Number of 'features' or channels used by the discriminator network in last layer.

        Returns:
            None
            
        """

        self.input_size = input_size
        self.nc = nc
        self.ndf = ndf
        self.num_layers = int(np.log(self.input_size)/np.log(2))

        self.layers = nn.Sequential()

        # add conv layer blocks: each block halves image size and doubles channels
        for i in range(self.num_layers-1):
            in_channels = nc if i==0 else ndf*2**(i-1)
            out_channels = ndf*2**i
            for layer in self.convblock(in_channels, out_channels):
                self.layers.append(layer)

        # last layer block generates probability
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(ndf*2**self.num_layers, 1))

    def convblock(self, in_channels, out_channels):
        out_size = int(self.ndf*2**(self.num_layers-1)/out_channels)
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                  nn.LayerNorm([out_channels, out_size, out_size]),
                  nn.LeakyReLU(True)]
        return layers

    def forward(self, input):
        output = self.layers(input)
        return output