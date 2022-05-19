import pandas as pd
import numpy as np
import cv2
import random
import itertools, os, time
import config
import matplotlib.pyplot as plt
from tacobox import Taco

class Sample:
    "sample from the dataset"
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath
        
class data_loader:
    def __init__(self, path, batch_size):
        self.batchSize = batch_size
        self.samples = []
        self.currIdx = 0
        self.charList = []
        
        # creating taco object for augmentation (checkout Easter2.0 paper)
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10
        )
        
        f = open(path + 'lines.txt')
        chars = set()
        for line in f:
            if not line or line[0]=='#':
                continue
            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9
            fileNameSplit = lineSplit[0].split('-')
            fileName = path + 'lines/' + fileNameSplit[0] + '/' +\
                       fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
            
            gtText = lineSplit[8].strip(" ").replace("|", " ")
            
            chars = chars.union(set(list(gtText)))
            self.samples.append(Sample(gtText, fileName))
        
        train_folders = [x.strip("\n") for x in open(path+"LWRT/train.uttlist").readlines()]
        validation_folders = [x.strip("\n") for x in open(path+"LWRT/validation.uttlist").readlines()]
        test_folders = [x.strip("\n") for x in open(path+"LWRT/test.uttlist").readlines()]

        self.trainSamples = []
        self.validationSamples = []
        self.testSamples = []

        for i in range(0, len(self.samples)):
            file = self.samples[i].filePath.split("/")[-1][:-4].strip(" ")
            folder = "-".join(file.split("-")[:-1])
            if (folder in train_folders): 
                self.trainSamples.append(self.samples[i])
            elif folder in validation_folders:
                self.validationSamples.append(self.samples[i])
            elif folder in test_folders:
                self.testSamples.append(self.samples[i])
        self.trainSet()
        self.charList = sorted(list(chars))
        
        
    def trainSet(self):
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples

    def validationSet(self):
        self.currIdx = 0
        self.samples = self.validationSamples
        
    def testSet(self):
        self.currIdx = 0
        self.samples = self.testSamples
        
    def getIteratorInfo(self):
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        return self.currIdx + self.batchSize <= len(self.samples)
    
    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)
            
        # scaling image [0, 1]
        img = img/255
        img = img.swapaxes(-2,-1)[...,::-1]
        target = np.ones((config.INPUT_WIDTH, config.INPUT_HEIGHT))
        new_x = config.INPUT_WIDTH/img.shape[0]
        new_y = config.INPUT_HEIGHT/img.shape[1]
        min_xy = min(new_x, new_y)
        new_x = int(img.shape[0]*min_xy)
        new_y = int(img.shape[1]*min_xy)
        img2 = cv2.resize(img, (new_y,new_x))
        target[:new_x,:new_y] = img2
        return 1 - (target)
    
    def apply_taco_augmentations(self, input_img):
        random_value = random.random()
        if random_value <= config.TACO_AUGMENTAION_FRACTION:
            augmented_img = self.mytaco.apply_vertical_taco(
                input_img, 
                corruption_type='random'
            )
        else:
            augmented_img = input_img
        return augmented_img

    def getNext(self, what='train'):
        while True:
            if ((self.currIdx + self.batchSize) <= len(self.samples)):
                
                itr = self.getIteratorInfo()
                batchRange = range(self.currIdx, self.currIdx + self.batchSize)
                if config.LONG_LINES:
                    random_batch_range = random.choices(range(0, len(self.samples)), k=self.batchSize)
                    
                gtTexts = np.ones([self.batchSize, config.OUTPUT_SHAPE])
                input_length = np.ones((self.batchSize,1))*config.OUTPUT_SHAPE
                label_length = np.zeros((self.batchSize,1))
                imgs = np.ones([self.batchSize, config.INPUT_WIDTH, config.INPUT_HEIGHT])
                j = 0;
                for ix, i in enumerate(batchRange):
                    img = cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        img = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
                    text = self.samples[i].gtText
                    
                    if config.LONG_LINES:
                        if random.random() <= config.LONG_LINES_FRACTION:
                            index = random_batch_range[ix]
                            img2 = cv2.imread(self.samples[index].filePath, cv2.IMREAD_GRAYSCALE)
                            if img2 is None:
                                img2 = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
                            text2 = self.samples[index].gtText
                            
                            avg_w = (img.shape[1] + img2.shape[1])//2
                            avg_h = (img.shape[0] + img2.shape[0])//2
                            
                            resized1 = cv2.resize(img, (avg_w, avg_h))
                            resized2 = cv2.resize(img2, (avg_w, avg_h))
                            space_width = random.randint(config.INPUT_HEIGHT//4, 2*config.INPUT_HEIGHT)
                            space = np.ones((avg_h, space_width))*255
                            
                            img = np.hstack([resized1, space, resized2])
                            text = text + " " + text2
                            
                    if len(self.samples) < 3000:# FOR VALIDATION AND TEST SETS
                        eraser=-1
                    img = self.preprocess(img)                    
                    imgs[j] = img
                    
                    val = list(map(lambda x: self.charList.index(x), text))
                    while len(val) < config.OUTPUT_SHAPE:
                        val.append(len(self.charList))
                        
                    gtTexts[j] = (val)
                    label_length[j] = len(text)
                    input_length[j] = config.OUTPUT_SHAPE
                    j = j + 1
                    if False:
                        plt.figure( figsize = (20, 20))
                        plt.imshow(img)
                        plt.show()
                        
                self.currIdx += self.batchSize
                inputs = {
                        'the_input': imgs,
                        'the_labels': gtTexts,
                        'input_length': input_length,
                        'label_length': label_length,
                }
                outputs = {'ctc': np.zeros([self.batchSize])}
                yield (inputs,outputs)
            else:
                self.currIdx = 0
                
    def getValidationImage(self):
        batchRange = range(0, len(self.samples))
        imgs = []
        texts = []
        reals = []
        for i in batchRange:
            img1 = cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE)
            real = cv2.imread(self.samples[i].filePath)
            if img1 is None:
                img1 = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
            img = self.preprocess(img1, augment=False)
            img = np.expand_dims(img,  0)
            text = self.samples[i].gtText
            imgs.append(img)
            texts.append(text)
            reals.append(real)
        self.currIdx += self.batchSize
        return imgs,texts,reals
    
    def getTestImage(self):
        batchRange = range(0, len(self.samples))
        imgs = []
        texts = []
        reals = []
        for i in batchRange:
            img1 = cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE)
            real = cv2.imread(self.samples[i].filePath)
            if img1 is None:
                img1 = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
            img = self.preprocess(img1, augment=False)
            img = np.expand_dims(img,  0)
            text = self.samples[i].gtText
            imgs.append(img)
            texts.append(text)
            reals.append(real)
        self.currIdx += self.batchSize
        return imgs,texts,reals