
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def linear(in_channels, out_channels, batch_norm=True):
    layers = []
    layers.append(nn.Linear(in_channels, out_channels, bias = False))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()
        # self.conv1 = conv(3, conv_dim, 4) 
        # self.conv2 = conv(conv_dim, conv_dim*2, 4)
        # self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

        # self.resnet_block = ResnetBlock(conv_dim*4)
        # self.resnet_block1 = ResnetBlock(conv_dim*4)

        # self.deconv3 = deconv(conv_dim*4, conv_dim*2, 4)
        # self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
        # self.deconv1 = deconv(conv_dim, 3, 4)

        #new
        self.conv1 = conv(3, conv_dim, 4) 
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

        latent_vector = 400

        self.mean = linear(conv_dim * 4 * 16 * 16, latent_vector)
        self.noise = linear(conv_dim * 4 * 16 * 16, latent_vector)

        self.deconv_linear = linear(latent_vector, conv_dim * 4 * 16 * 16)

        self.deconv3 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv1 = deconv(conv_dim, 3, 4)

    def forward(self, x):
        # out = F.relu(self.conv1(x))            
        # out = F.relu(self.conv2(out))         
        # out = F.relu(self.conv3(out)) 

        # out = F.relu(self.resnet_block(out))  
        # out = F.relu(self.resnet_block1(out))   

        # out = F.relu(self.deconv3(out))        
        # out = F.tanh(self.deconv2(out))        
        # out = F.tanh(self.deconv1(out)) 
        # return out

        #new
        out = F.relu(self.conv1(x))            
        out = F.relu(self.conv2(out))         
        out = F.relu(self.conv3(out)).view(-1, 16 * 16 * 256)

        # print(out.size())

        logvar = self.noise(out)
        mu = self.noise(out)

        std = logvar.mul(0.5).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        out = eps.mul(std).add(mu)

        out = F.relu(self.deconv_linear(out)).view(-1, 256, 16, 16)

        out = F.relu(self.deconv3(out))       
        out = F.tanh(self.deconv2(out))        
        out = F.tanh(self.deconv1(out)) 
        return out, logvar, mu


class DCDiscriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(DCDiscriminator, self).__init__()

        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4)
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4)
        self.conv4 = conv(in_channels=conv_dim * 4, out_channels=1, kernel_size=4, stride=1, padding=0,
                          batch_norm=False)

    def forward(self, x):

        out = F.relu(self.conv1(x))    # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))  # BS x 128 x 8 x 8
        out = F.relu(self.conv3(out))  # BS x 256 x 4 x 4

        out = self.conv4(out).squeeze()
        out = F.sigmoid(out)
        return out
