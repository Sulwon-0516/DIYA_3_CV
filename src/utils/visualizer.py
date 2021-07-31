'''
Loss function plotter
'''

import visdom
import torch
import numpy as np

''' ref : https://github.com/noagarcia/visdom-tutorial with some modification'''
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        try:
            self.viz = visdom.Visdom(env = env_name)
        except ConnectionRefusedError:
            print("please turn on Visdom server.\nuse python -m visdom.server")
            assert(0)
        self.env = env_name
        self.figs = {}
        
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.figs:
            self.figs[var_name] = self.viz.line(
                X = x, 
                Y = y.unsqueeze(0), 
                env = self.env, 
                opts = dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel='Steps',
                    ylabel=var_name,
                    )
                )
        else:
            self.viz.line(
                X = x.unsqueeze(0), 
                Y = y.unsqueeze(0), 
                env = self.env, 
                win = self.figs[var_name], 
                update = 'append',
                opts = dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel='Steps',
                    ylabel=var_name,
                    )
                )
            
            
class VisLine(object):
    def __init__(self):
        self.visline = VisdomLinePlotter("graph")
    
        self.visline.plot(
            var_name= "train_loss_step",
            split_name= "train_loss",
            title_name= "loss per step",
            x = torch.zeros(1),
            y = torch.zeros(1),
        )
        
        self.visline.plot(
            var_name= "train_loss_epoch",
            split_name= "train_loss",
            title_name= "loss per epoch",
            x = torch.zeros(1),
            y = torch.zeros(1),
        )
        
        self.visline.plot(
            var_name= "train_acc_epoch",
            split_name= "train_acc",
            title_name= "acc per epoch(train)",
            x = torch.zeros(1),
            y = torch.zeros(1),
        )
        
        self.visline.plot(
            var_name= "val_loss_epoch",
            split_name= "val_loss",
            title_name= "val loss per epoch",
            x = torch.zeros(1),
            y = torch.zeros(1),
        )
        
        self.visline.plot(
            var_name= "val_acc_epoch",
            split_name= "val_acc",
            title_name= "acc per epoch(val)",
            x = torch.zeros(1),
            y = torch.zeros(1),
        )