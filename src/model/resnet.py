'''
This code is not yet generalized

In this code, I change the head of NN with different CNN, to avoid the dimension
reduce larger than my expectation.
'''

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
from torchsummary import summary 

debug_opt = False

class resnet(nn.Module):
    def __init__(self,num_depth = 50, input_dim = [28, 28]):
        super(resnet, self).__init__()
        '''
        input :
        - num_depth : 18, 34, 50, 101, 152
        - input_dim : input image dimension
        '''
        possible_depth = [18, 34, 50, 101, 152]
        if num_depth not in possible_depth:
            print("%d is wrong resnet size"%(num_depth))
            assert(0)
            
        if num_depth == 50:
            backbone = torchvision.models.resnet50()
        # as exec doesn't work, i gave up automation
        
    
        if debug_opt:
            print(list(backbone.children()))
        self.input = nn.Conv2d(in_channels=1, out_channels = 512, kernel_size = 1)
        self.backbone = nn.Sequential(*(list(backbone.children())[6:-1]))
        self.output = nn.Linear(2048,10)
        
    def forward(self, x):
        out = self.input(x)
        out = self.backbone(out)
        out = torch.reshape(out,(out.shape[0],-1))
        out = self.output(out)
        return out
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = resnet().to(device)
    summary(model, (1,28,28))