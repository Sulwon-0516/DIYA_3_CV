import torch
import PIL
import os
import numpy as np
import random
import cv2
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

f_name_opt = "82"

class data_gen():
    def __init__(self, is_train = True):
        train_converter = {2: lambda s: (ord(s)-ord('A'))}
        test_converter = {1: lambda s: (ord(s)-ord('A'))}
        

        self.is_train = is_train
        if is_train:
            path = TRAIN_PATH
            conv = train_converter
        else:
            path = TEST_PATH
            conv = test_converter
        
        self.data = np.loadtxt(
            path, 
            delimiter = ',', 
            dtype = np.int32,
            skiprows=1, 
            converters= conv
            )
        self.data = np.transpose(self.data)

        self.id = self.data[0]
        if self.is_train:
            self.digit = self.data[1]
            self.letter = self.data[2]
            self.input = np.transpose(self.data[3:])
        else:
            self.letter = self.data[1]
            self.input = np.transpose(self.data[2:])

        if debug_opt:
            print(self.id[0:10])
            print(self.digit[0:10])
            print(self.letter[0:10])
            print(self.input[0].shape)

        self.mean = np.mean(self.input)
        self.std = np.std(self.input)
        
        # define normalize transformer
        self.trans = transforms.Normalize((0.1307,), (0.3081,))
        
        
        (
            self.avg_digits, 
            self.num_letters,
            self.letter_ids
            ) = self.get_avg_imgs()
        

    def get_random_mask(self, letter, thrs = 10):
        r_ind = random.randrange(1, self.num_letters[letter] + 1)
        ind = self.letter_ids[letter][r_ind]
        
        img = self.input[int(ind)]
        masked_img = img
        masked_img[masked_img<=thrs] = 0
        masked_img[masked_img>thrs] = 1
        
        if np.mean(masked_img) == 0:
            print("wrong")
        
        
        return masked_img
        
    def generate_img(self, letter, digit, is_aug = False):
        # mask : around 70
        # org : around 70
        mask = self.get_random_mask(letter)
        org_img = self.avg_digits[digit]
        
        #Implement Random Shift here.
        masked_img = org_img * mask

        # Current, Simply Add
        blur_mask = blur(mask)
        gen_img = masked_img/2 + blur_mask[:,0]*60 
        #print(masked_img)
        #print(mask)
        
        return gen_img
        #image_plot(gen_img, digit, letter, -2)
        

    def get_avg_imgs(self):
        if self.is_train:
            avg_digits = np.zeros((10,784))
            num_digits = np.zeros(10)
        avg_letters = np.zeros((26,784))
        num_letters = np.zeros(26)
        
        
        letter_ids = [np.zeros(1)]
        for i in range(25):
            letter_ids.append(np.zeros(1))
            
            
        for i in range(len(self.letter)):
            avg_letters[int(self.letter[i])] += self.input[i]
            num_letters[int(self.letter[i])] += 1
            letter_ids[int(self.letter[i])] = np.append(letter_ids[int(self.letter[i])], [i])
            
            if self.is_train:
                avg_digits[int(self.digit[i])] += self.input[i]
                num_digits[int(self.digit[i])] += 1
        
        
        plt.figure(1)
        plt.title("letters plotting")
        plt.subplot(4,7,1)
        for i in range(26):
            plt.subplot(4,7,1+i)
            avg_letters[i]/= num_letters[i]
            img = np.reshape(avg_letters[i],(28,28))
            plt.imshow(img)
            #image_plot(avg_letters[i], 0, i, 0)
        f_name = os.path.join(IMG_PATH,"letters_avg"+".png" )
        plt.savefig(f_name)
        
        if self.is_train:
            plt.figure(2)
            plt.title("digits plotting")
            plt.subplot(2,5,1)
            for i in range(10):
                plt.subplot(2,5,1+i)
                avg_digits[i]/= num_digits[i]
                img = np.reshape(avg_digits[i],(28,28))
                img = sharpen(img)
                avg_digits[i] = np.reshape(img,(784))
                plt.imshow(img)
                #image_plot(avg_digits[i], i, 0, -1)
            f_name = os.path.join(IMG_PATH,"filt_digits_avg"+".png" )
            plt.savefig(f_name)

        return avg_digits, num_letters, letter_ids
    
    
    def test_gen(self):
        for i in range(10):
            plt.figure(0)
            plt.title("generated img, num :" + str(i))
            plt.subplot(4,7,1)
            for j in range(26):
                img = A.generate_img(j, i)
                plt.subplot(4,7,1+j)
                img = np.reshape(img,(28,28))
                plt.imshow(img)
            f_name = os.path.join(IMG_PATH,"gen_imgs_"+str(i)+".png" )
            plt.savefig(f_name)
            

def blur(img):
    img = img.astype(np.float32)
    kernel = np.ones((3,3),np.float32)/(3*3)
    blur = cv2.filter2D(img,-1,kernel)
    
    return blur          

def sharpen(img):
    if True:
        img[img<65] = 0
    
    res = img.astype(np.float32)
    kernel= np.array([[0, -1, 0],[-1, 8, -1],[0, -1, 0]], np.float32)
    kernel= 1/4 * kernel
    for i in range(1):
        res = cv2.filter2D(res, -1, kernel)
        res[res<60]=0

    
    res[res<0] = 0
    if False:
        res[res<60] = 0
    
    return res



def image_plot(input, digit, letter, idx):
    '''
    input : flatten 784 length numpy array.
    print image 
    '''
    title = str(idx)+"_digit :"+str(digit)+ ",letter :"+chr(ord('A')+letter)
    img = np.reshape(input,(28,28))
    plt.figure(0)
    if not no_monitor:
        plt.imshow(img)
    else:
        f_name = os.path.join(IMG_PATH, title+".png")
        plt.imsave(f_name, img)
    plt.title(title)


if __name__ == "__main__":
    A = data_gen()
    #A.get_avg_imgs()
    
    A.test_gen()
    
    #print("test dataset mean :"+str(B.mean)+",std :"+str(B.std))
