import torch
import os
import csv
from torch.utils.data import Dataset
import numpy as np

import random

debug_opt = True
no_monitor = True

DATASET_PATH = os.path.join(os.path.dirname(__file__),"../../datasets/")
TRAIN_PATH = os.path.join(DATASET_PATH, "train.csv")
TEST_PATH = os.path.join(DATASET_PATH, "test.csv")

VAL_PATH = os.path.join(DATASET_PATH,"splitted/82val.csv")
TRA_PATH = os.path.join(DATASET_PATH,"splitted/82train.csv")

def train_split(split_ratio = [0.2, 0.8]):
    train_converter = {2: lambda s: (ord(s)-ord('A'))}
    test_converter = {1: lambda s: (ord(s)-ord('A'))}


    data = np.loadtxt(
        TRAIN_PATH, 
        delimiter = ',', 
        dtype = np.int32,
        skiprows=1, 
        converters= train_converter
        )
    data = np.transpose(data)

    id = data[0]
    digit = data[1]
    letter = data[2]
    input = np.transpose(data[3:])

    len_data = len(id)
    
    num_map = np.zeros((10,26))
    
    # get the statistic
    for i in range(len(id)):
        num_map[
            int(digit[i]),
            int(letter[i])
            ]+= 1

    
        
    
    val_id = []
    train_id = []
    
    train_num_map = np.zeros((10,26))
    
    for i in range(len(id)):
        #Thinks without shuffling.
        #Or I can shuffle it by shuffle the "id"
        # -> but it's little bit complex.
        
        
        if train_num_map[
            int(digit[i]),
            int(letter[i])
            ] == 0:
            # First, save it to train_id.
            train_num_map[int(digit[i]), int(letter[i])] += 1
            train_id.append(id[i])
        else:
            # Second, randomly select one.
            ran = np.random.binomial(1,split_ratio[0])
            if ran < 0.5:
                train_num_map[int(digit[i]), int(letter[i])] += 1
                train_id.append(id[i])
            else:
                val_id.append(id[i])
        
    with open(VAL_PATH, 'w', newline='') as valfile:
        valfile.write('id,digit,letter')
        for i in range(28*28):
            valfile.write(','+str(i))
        valfile.write('\n')
        
        for i in val_id:
            valfile.write(str(i))
            valfile.write(','+str(digit[i-1]))
            valfile.write(','+chr(ord('A')+letter[i-1]))
            for j in range(28*28):
                valfile.write(','+str(input[i-1][j]))
                # it call sys-call 700 times so it would be slow
            valfile.write('\n')
            
    with open(TRA_PATH, 'w', newline='') as trainfile:
        trainfile.write('id,digit,letter')
        for i in range(28*28):
            trainfile.write(','+str(i))
        trainfile.write('\n')
        
        for i in train_id:
            trainfile.write(str(i))
            trainfile.write(','+str(digit[i-1]))
            trainfile.write(','+chr(ord('A')+letter[i-1]))
            for j in range(28*28):
                trainfile.write(','+str(input[i-1][j]))
                # it call sys-call 700 times so it would be slow
            trainfile.write('\n')
        
    if debug_opt:
        print(id[0:10])
        print(digit[0:10])
        print(letter[0:10])
        print(input[0].shape)
        print(len(val_id))
        print(len(train_id))
        

if __name__ == '__main__':
    train_split()