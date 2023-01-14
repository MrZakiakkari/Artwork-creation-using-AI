import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


class Discriminator(nn.Module):
    def _init_(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128) ,
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1), 
            nn.Sigmoid(),
        )
        
        def forward(self, x):
            return self.disc(x)
        
        class Generator(nn.Module):
            def _init_(self, z_dim, img_dim):
                super()._init_()
                self.gen = nn.Sequential(
                    nn.Linear(z_dim, 256),
                    nn.LeakyRelu(0.1),
                    nn.Linear(256, img_dim),
                    nn.Tahn(),
                )
            def forward(self, x):
                return self.gen(x) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64
    image_dim = 28 * 28 *1    