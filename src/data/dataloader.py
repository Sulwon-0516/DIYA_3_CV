import torch
import PIL
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img

from torchvision import transforms

debug_opt = False
no_monitor = True

DATASET_PATH = os.path.join(os.path.dirname(__file__),"../../datasets/")
TRAIN_PATH = os.path.join(DATASET_PATH, "train.csv")
TEST_PATH = os.path.join(DATASET_PATH, "test.csv")

IMG_PATH = os.path.join(DATASET_PATH, "images/")
GRAPH_PATH = os.path.join(DATASET_PATH, "graphs/")

VAL_PATH = os.path.join(DATASET_PATH,"splitted/82val.csv")
TRA_PATH = os.path.join(DATASET_PATH,"splitted/82train.csv")
GEN_PATH = os.path.join(DATASET_PATH,"generated/gen_100.csv")

global f_name_opt
f_name_opt = "gen"

class my_dataloader(Dataset):
    def __init__(self, is_train, is_val, is_gen = False):
        self.is_train = is_train
        self.is_val = is_val
        self.is_gen = is_gen
        train_converter = {2: lambda s: (ord(s)-ord('A'))}
        test_converter = {1: lambda s: (ord(s)-ord('A'))}

        if is_gen:
            path = GEN_PATH
        elif is_val:
            path = VAL_PATH
        else:
            path = TRAIN_PATH   ####### changed 
        

        if is_train:
            self.data = np.loadtxt(
                path, 
                delimiter = ',', 
                dtype = np.int32,
                skiprows=1, 
                converters= train_converter
                )
            self.data = np.transpose(self.data)

            self.id = self.data[0]
            self.digit = self.data[1]
            self.letter = self.data[2]
            self.input = np.transpose(self.data[3:])

            if debug_opt:
                print(self.id[0:10])
                print(self.digit[0:10])
                print(self.letter[0:10])
                print(self.input[0].shape)

        else:
            self.data = np.loadtxt(
                TEST_PATH,
                delimiter = ',', 
                dtype = np.int32,
                skiprows=1,
                converters= test_converter)
            self.data = np.transpose(self.data)

            self.id = self.data[0]
            self.letter = self.data[1]
            self.input = np.transpose(self.data[2:])

            if debug_opt:
                print(self.id[0:10])
                print(self.letter[0:10])
                print(self.input[0].shape)


        self.mean = np.mean(self.input)
        self.std = np.std(self.input)
        
        # define normalize transformer
        self.trans = transforms.Normalize((0.1307,), (0.3081,))


    def __getitem__(self, idx):
        _input = torch.from_numpy(np.reshape(self.input[idx],(28,28))).unsqueeze(0)/255
        if self.is_train:
            digit = torch.from_numpy(np.asarray(self.digit[idx]))
        letter = torch.from_numpy(np.asarray(self.letter[idx]))
        _id = torch.from_numpy(np.asarray(self.id[idx]))
        
        if debug_opt:
            print(_input*256)
            image_plot(self.input[idx], self.digit[idx], self.letter[idx], idx)

        if debug_opt:
            print(_input.shape)  
            print(digit.shape)
            
        if self.is_gen:                 ####### gerated of GAN (remove it after GAN training)
            return _input, digit, _id

        if self.is_train:
            return self.trans(_input), digit, _id

        else:
            return self.trans(_input), _id

    def __len__(self):
        return len(self.letter)


    def statistics(self):
        
        num_letters = np.zeros(26) 
        if self.is_train:
            num_map = np.zeros((10,26))
            num_digit = np.zeros(10)   
        pixel_statistics = np.zeros(256)

        for i in range(len(self.letter)):
            num_letters[int(self.letter[i])] += 1
            if self.is_train:
                num_map[
                    int(self.digit[i]),
                    int(self.letter[i])
                    ]+= 1
                num_digit[int(self.digit[i])] += 1
                
            for j in range(28*28):
                pixel_statistics[int(self.input[i][j])] +=1
        
        if self.is_train:
            print("train")
            print("----------------------------------")
            print("digit statistic")
            print(num_digit)
            print("letter statistic")
            print(num_letters)
            
            '''
            plt.figure(0)
            plt.subplot(1,3,1)
            plt.bar(np.arange(10),num_digit)
            plt.title("digit hist")
            plt.subplot(1,3,2)
            plt.bar(np.arange(26),num_letters)
            plt.title("letter hist")
            plt.subplot(1,3,3)
            plt.bar(np.arange(256),pixel_statistics)
            plt.title("pixel hist")
            '''
            
            for i in range(3):
                
                if i == 0:
                    plt.figure(0)
                    plt.clf()
                    plt.bar(np.arange(10),num_digit)
                    title = "digit hist"
                    plt.title(title)
                    plt.xlabel("digit")
                elif i==1:
                    plt.figure(0)
                    plt.clf()
                    plt.bar(np.arange(26),num_letters)
                    title = "letter hist"
                    plt.title(title)
                    plt.xlabel("letter")
                    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                               labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
                else:
                    plt.figure(0)
                    plt.clf()
                    plt.bar(np.arange(256),pixel_statistics)
                    title = "pixel hist"
                    plt.title(title)
                    
                if not no_monitor:
                    plt.show()
                else:
                    f_name = f_name_opt + title + ".png"
                    if self.is_val:
                        f_name = 'val_'+f_name
                    f_name = os.path.join(GRAPH_PATH, f_name)
                    
                    plt.savefig(f_name)
                
            
        
            plt.figure(1)
            plt.bar(np.arange(256)[5:],pixel_statistics[5:])
            if not no_monitor:
                plt.show()
            else:
                f_name = f_name_opt + "train_pixel_stats.png"
                if self.is_val:
                    f_name = 'val_'+f_name
                f_name = os.path.join(GRAPH_PATH, f_name)
                plt.savefig(f_name)
                
            plt.clf()
            fig, ax = plt.subplots(figsize=(16,16))
            im = ax.imshow(num_map)
            
            for i in range(10):
                for j in range(26):
                    text = ax.text(j, i, num_map[i, j],
                        ha="center", va="center", color="w")
            if not no_monitor:
                plt.show()
            else:
                f_name = f_name_opt + "train_heat_map.png"
                if self.is_val:
                    f_name = 'val_'+f_name
                f_name = os.path.join(GRAPH_PATH, f_name)
                plt.savefig(f_name)
            
            

        else:
            print("test")
            print("----------------------------------")
            print("letter statistic")
            print(num_letters)
            '''
            plt.figure(2)
            plt.subplot(1,2,1)
            plt.bar(np.arange(26),num_letters)
            plt.title("letter hist")
            plt.subplot(1,2,2)
            plt.bar(np.arange(256),pixel_statistics)
            plt.title("pixel hist")
            '''
            for i in range(2):
                if i==0:
                    plt.figure(2)
                    plt.clf()
                    plt.bar(np.arange(26),num_letters)  
                    
                    params = {'xtick.labelsize':'large',
                              'ytick.labelsize':'large',
                              'axes.titlesize':35,
                              'axes.labelsize':35
                    }
                    plt.rcParams.update(params)
                    
                    
                    title = "letter hist"
                    plt.title(title)
                    plt.xlabel("letter")
                    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                               labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
                    plt.yticks([0,200,400,600,800])
                    
                    axes = plt.gca()
                    axes.xaxis.label.set_size(30)
                    axes.yaxis.label.set_size(30)
                    
                   
                else:
                    plt.figure(2)
                    plt.clf()
                    plt.bar(np.arange(256),pixel_statistics)
                    title = "pixel hist"
                    plt.title(title)
                
                if not no_monitor:
                    plt.show()
                else:
                    f_name = os.path.join(GRAPH_PATH, f_name_opt + title + ".png")
                    plt.savefig(f_name)
        
            plt.figure(3)
            plt.bar(np.arange(256)[5:],pixel_statistics[5:])
            if not no_monitor:
                plt.show()
            else:
                f_name = os.path.join(GRAPH_PATH, f_name_opt + "test_pixel_stats.png")
                plt.savefig(f_name)
                

        


def image_plot(input, digit, letter, idx):
    '''
    input : flatten 784 length numpy array.
    print image 
    '''
    title = "digit :"+str(digit)+ ",letter :"+chr(ord('A')+letter)+"_"+str(idx)
    img = np.reshape(input,(28,28))
    plt.figure()
    if not no_monitor:
        plt.imshow(img)
    else:
        f_name = os.path.join(IMG_PATH, title+".png")
        plt.imsave(f_name, img)
    plt.title(title)

if __name__ == "__main__":
    train_data = my_dataloader(False,False,False)
    val_data = my_dataloader(True, False,False)
    A = my_dataloader(True,False)
    
    f_name_opt = "train"
    val_data.statistics()
    f_name_opt = "test"
    train_data.statistics()

    print("train dataset mean :"+str(A.mean)+",std :"+str(A.std))
    #print("test dataset mean :"+str(B.mean)+",std :"+str(B.std))
