import _init_paths

import os
import torchvision
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict

from utils.visualizer import VisdomLinePlotter, VisLine
from data.dataloader import my_dataloader
from model.resnet import resnet
from _train import _train
from _val import _val

RES_PATH = os.path.join(os.path.dirname(__file__),"../../checkpoint/")
g_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def test(model = None, device = g_device):
    if model == None:
        model = resnet()
        model.to(device)
    
    test_data = my_dataloader(False,False)
    test_dataloader = DataLoader(
        dataset = test_data, 
        shuffle = False,
        batch_size = 2048)
    
    fname = "result.csv"
    fname = os.path.join(RES_PATH, fname)
    resfile = open(fname, 'w', newline='')
    resfile.write('id,digit')
    resfile.write('\n')
    
    model.eval()
    with torch.no_grad():
        tot_loss = 0
        for i, data in enumerate(test_dataloader):
            input, ids = data
            input = input.to(device)

            output = model(input)
            
            _, ind = torch.max(output.detach().cpu(),axis = 1)
            
            for j in range(len(ids)):
                resfile.write(str(ids[j].item())+','+str(ind[j].item())+'\n')
            
            
    
    #print("tot loss of test :",tot_loss)
    
if __name__ == '__main__':
    test()