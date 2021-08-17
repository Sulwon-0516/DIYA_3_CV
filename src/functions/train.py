import _init_paths

#import torchvision
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
from test import test

Multi_GPU = False

CFG = edict()
CFG.model = "resnet"
CFG.model_size = 50
CFG.input_dim = [28,28]

CFG.EPOCH = 200
CFG.batch_size = 96

CFG.LR = 5e-4
CFG.weight_decay = 1e-4
CFG.LR_milestones = [210]
CFG.LR_decayrate = 0.1

CFG.load_prev = False
CFG.inference_bsize = 128



g_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train(device = g_device, train_cfg = CFG, multi_gpu = Multi_GPU):
    
    # define the model.
    model = model_select(train_cfg)
    model.to(device)
    
    
    
    # define dataloader
    train_data = my_dataloader(True,False,True)
    train_dataloader = DataLoader(
        dataset = train_data, 
        shuffle = True,
        batch_size = train_cfg.batch_size)
    
    val_data = my_dataloader(True, False,False)
    val_dataloader = DataLoader(
        dataset = val_data, 
        shuffle = False,
        batch_size = train_cfg.inference_bsize)
    
    # define optimizer, scheduler and critertion
    crit = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr = train_cfg.LR,
        weight_decay = train_cfg.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones = train_cfg.LR_milestones, 
        gamma = train_cfg.LR_decayrate,
        )
    
    # parallel gpus
    if multi_gpu:
        print("using %d GPUs for training" % torch.cuda.device_count())
        model = nn.DataParallel(model)
        
    # visualizer
    vl = VisLine()
        
    # train epoch
    step = 0
    for i in range(train_cfg.EPOCH):
        stat = epoch_stat()
        val_stat = epoch_stat()
        
        (
            tot_loss,
            step,
        ) = _train(
            model = model,
            dataloader = train_dataloader,
            crit = crit,
            opt = optimizer,
            device = device,
            stat = stat,
            vis = vl,
            step = step,
        )
        
        _val(
            model = model,
            dataloader = val_dataloader,
            crit = crit,
            device = device,
            stat = val_stat
        )

        avg_loss, acc = stat.result()
        print("epoch : %d, "%(i),acc,avg_loss)
        vl.visline.plot(
            var_name= "train_loss_epoch",
            split_name= "train_loss",
            title_name= "loss per epoch",
            x = torch.tensor(i),
            y = avg_loss,
        )
        
        vl.visline.plot(
            var_name= "train_acc_epoch",
            split_name= "train_acc",
            title_name= "acc per epoch(train)",
            x = torch.tensor(i),
            y = acc,
        )

        avg_loss, acc = val_stat.result()
        print("val epoch : %d, "%(i),acc,avg_loss)
        vl.visline.plot(
            var_name= "val_loss_epoch",
            split_name= "val_loss",
            title_name= "val loss per epoch",
            x = torch.tensor(i),
            y = avg_loss,
        )
        
        vl.visline.plot(
            var_name= "val_acc_epoch",
            split_name= "val_acc",
            title_name= "acc per epoch(val)",
            x = torch.tensor(i),
            y = acc,
        )
        
        scheduler.step()
    torch.save({
        'model_state_dict' : model.state_dict()
        },'./temp_2.pt')
    
    return model


class epoch_stat():
    def __init__(self):
        self.loss = 0
        self.num_true = 0
        self.num_tested = 0

    def step(self, output, digit, loss):
        _, ind = torch.max(output.detach().cpu(),axis = 1)
        digit = digit.cpu()

        self.num_true += torch.sum(ind==digit)
        self.num_tested += len(digit)
        self.loss += loss.detach().cpu()*len(digit)
    
    def result(self):
        avg_loss = self.loss/self.num_tested
        acc = self.num_true/self.num_tested 

        return avg_loss, acc

def model_select(cfg):
    if cfg.model == 'resnet':
        model = resnet(cfg.model_size, cfg.input_dim)
    else:
        print("model named '%s' isn't prepared"%cfg.model)
        assert(0)
    return model

if __name__=="__main__":
    model = train()
    #test(model)