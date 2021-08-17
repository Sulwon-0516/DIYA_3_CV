import os.path 
import sys

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir,'..')
vis_path = os.path.join(this_dir,'../../utils')

if lib_path not in sys.path:
    sys.path.insert(0,lib_path)
    
if vis_path not in sys.path:
    sys.path.insert(0,vis_path)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from dataloader import my_dataloader
from visualizer import VisdomLinePlotter


# This model is from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py
# Only modified the smal l part of this code.

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 32   
sample_dir = os.path.join(this_dir,'samples')

DATASET_PATH = os.path.join(os.path.dirname(__file__),"../../../datasets/")
TRAIN_PATH = os.path.join(DATASET_PATH, "train.csv")

vl = VisdomLinePlotter()

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


org_data = my_dataloader(True, False,False)
gen_data = my_dataloader(True,False,True)


# Data loader
data_loader = torch.utils.data.DataLoader(dataset=org_data,
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          drop_last=True)
gen_loader = torch.utils.data.DataLoader(dataset=gen_data,
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          drop_last=True)

class Get_gen(object):
    def __init__(self, gen_loader):
        super(Get_gen,self).__init__()
        self.loader = gen_loader
        self.enumer = enumerate(self.loader)
        
    def get(self):
        try:
            _, (ret,_,_) = next(self.enumer)
        except StopIteration:
            self.enumer = enumerate(self.loader)
            _, (ret,_,_)  = next(self.enumer)
        return ret
    
GEN = Get_gen(gen_loader)
'''
old disc (MLP based implementation)
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.BatchNorm2d(),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

old gen (MLP based implementation)

# Generator 
class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__()
        self.trans = transforms.Normalize((0.1307,), (0.3081,))
        self.l1 = nn.Linear(image_size + latent_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, image_size)

    def forward(self, noise, x):
        trans_x = x.clone().detach()
        trans_x = trans_x.view(-1,1,28,28)
        trans_x = self.trans(trans_x)
        trans_x = trans_x.view(-1,image_size)
            
        input = torch.cat((noise,trans_x),1)
        out = F.relu(self.l1(input))
        out = F.relu(self.l2(out))
        out = F.tanh(self.l3(out))
        
        out = out + x
        
        out = out.view(-1,1,28,28)
        out = self.trans(out)
        out = out.view(-1,image_size)
        
        return out
'''

# Discriminator
class Disc(nn.Module):
    def __init__(self):
        super(Disc,self).__init__()
        
        self.D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(2,2), bias = False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(2,2), stride = 2,  bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(2,2), stride = 2, bias = False),
            nn.BatchNorm2d(64)
        )
        self.last = nn.Linear(64*7*7, 1)
        self.sig = nn.Sigmoid()
    
    def forward(self, input):
        input = input.view(-1,1,28,28)
        out = self.D(input)
        out = out.view(-1,7*7*64)
        out = self.last(out)
        out = out.view(out.shape[0],-1)
        out = self.sig(out)
        
        return out
        
        
# Generator 
class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__()
        self.trans = transforms.Normalize((0.1307,), (0.3081,))
        self.l1 = nn.Linear(image_size + latent_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, image_size)

    def forward(self, noise, x):
        trans_x = x.clone().detach()
        trans_x = trans_x.view(-1,1,28,28)
        trans_x = self.trans(trans_x)
        trans_x = trans_x.view(-1,image_size)
            
        input = torch.cat((noise,trans_x),1)
        out = F.relu(self.l1(input))
        out = F.relu(self.l2(out))
        out = F.tanh(self.l3(out))
        
        out = out + x
        
        out = out.view(-1,1,28,28)
        out = self.trans(out)
        out = out.view(-1,image_size)
        
        return out
        
G = Gen()
D = Disc()
# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

step = 0
# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        gen_datas= GEN.get()
        gen_datas = gen_datas.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z, gen_datas)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z, gen_datas)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 30 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
        
        step += 1
        #Implement visualizer here
        vl.plot(
            var_name= "disc_loss_step",
            split_name= "disc_loss",
            title_name= "disc_loss",
            x = torch.tensor(step),
            y = d_loss.detach().cpu(),
        )
        vl.plot(
            var_name= "gen_loss_step",
            split_name= "gen_loss",
            title_name= "gen_loss",
            x = torch.tensor(step),
            y = g_loss.detach().cpu(),
        )
        
        vl.plot(
            var_name= "real_score_step",
            split_name= "real_score",
            title_name= "real_score",
            x = torch.tensor(step),
            y = real_score.detach().cpu().mean(),
        )
        
        vl.plot(
            var_name= "fake_score_step",
            split_name= "fake_score",
            title_name= "fake_score",
            x = torch.tensor(step),
            y = fake_score.detach().cpu().mean(),
        )
        
        
        
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

    if (epoch % 10) == 0:   # Save the model checkpoints 
        torch.save(G.state_dict(), 'G.ckpt')
        torch.save(D.state_dict(), 'D.ckpt')
        
    print("cur epoch : ",epoch)