
import os
import torch
import matplotlib.pyplot as plt

import json
import numpy as np

SAMPLE_PATH = "./sample"
RES_PATH = "./result"
RES_FILE = "result_1.json"


def save_model(path,
               model, 
               optimizer,
               scheduler,
               tot_loss,
               epoch,
               tot_EPOCH,
               num_tot_data,
               F_NAME,
               MODEL_NAME,
               is_best = False):
    if is_best:
        f_name = os.path.join(path,F_NAME%(MODEL_NAME,num_tot_data))
    else:
        f_name = os.path.join(path,F_NAME%(MODEL_NAME,num_tot_data,tot_loss,epoch,tot_EPOCH))
    
    Path(path).mkdir(exist_ok = True)

    if isinstance(model, torch.nn.DataParallel):
        to_save = model.module.state_dict()
    else:
        to_save = model.state_dict()


    if scheduler == None:
        torch.save({'epoch': epoch, 
            'model' : MODEL_NAME,
            'model_state_dict': to_save, 
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : None,
            'loss':tot_loss,
            'max_epoch' : tot_EPOCH
            },f_name)
        print("saved model to",f_name)
    else:
        torch.save({'epoch': epoch, 
            'model' : MODEL_NAME,
            'model_state_dict': to_save, 
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'loss':tot_loss,
            'max_epoch' : tot_EPOCH
            },f_name)
        print("saved model to",f_name)

def load_model(path,file,model,optimizer,scheduler,config = None, IS_TRAIN = False):
    if not os.path.isdir(path):
        print("Wrong model directory {}".format(path))
        assert(0)
    f_path = os.path.join(path,file)
    if not os.path.isfile(f_path):
        print("There isn't file saved model file %s in %s"%(file,path))
        assert(0)
    checkpoint = torch.load(f_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if IS_TRAIN:
        if config is not None and config.LOAD_PREV_OPTIM:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        if scheduler != None and config.LOAD_PREV_SCHED:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            last_epoch = checkpoint['scheduler_state_dict']['last_epoch']
            scheduler.step(epoch = last_epoch)
        model.train()
    else:
        start_epoch = 0
        loss = 0
        model.eval()
    print('loaded model :',checkpoint['model'])
    return model, optimizer, scheduler, start_epoch, loss.cpu()

#simple coco testing tools with printing result.

def load_pretrain_model(file,path,model):
    if not os.path.isdir(path):
        print("Wrong model directory {}".format(path))
        assert(0)
    f_path = os.path.join(path,file)
    if not os.path.isfile(f_path):
        print("There isn't file saved model file %s in %s"%(file,path))
        assert(0)
    checkpoint = torch.load(f_path)
    pre_state_dict = checkpoint['model_state_dict']
    
    model_dict = model.state_dict()
    pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in model_dict}
    model_dict.update(pre_state_dict)
    model.load_state_dict(model_dict)
    return model

