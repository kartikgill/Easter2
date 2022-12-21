## predict_test.py; which is made for single line images

from data_loader import data_loader
import config
from predict import load_easter_model
import cv2
import numpy as np
import itertools
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", help="A path of the image file")
args = parser.parse_args()

def preprocess(img):
            
    # preprocessing is needed 
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

def decoder(output,letters):
    ret = []
    for j in range(output.shape[0]):
        out_best = list(np.argmax(output[j,:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

## rewritten class file for efficient prediction

class infer_img:
    
    def __init__(self, charlist):
        
        self.model = load_easter_model(config.BEST_MODEL_PATH)
        self.charlist = charlist
        
    def predict(self, img_path):
        
        img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = preprocess(img1)
        img = np.expand_dims(img,  0)
        output = self.model.predict(img)
        prediction = decoder(output, self.charlist)
        
        return prediction[0]

if __name__ == "__main__":
    ## usage
    ## python predict_line.py --path ~/garbage/images/test1.jpg

    with open("../data/charlist") as f:
        charlist = [word.strip("\n") for word in f ]

    ## instead of loading the data path, you can write the charlist file in your local storage for the next time.
    infer_obj = infer_img(charlist)

    print(infer_obj.predict(args.path)) ## change the image path with the file path you want

''' **Note**: this prediction is only good with line level images because this model has trained on IAM dataset 
which contains only line level images. If you want to predict the real world images which contain multiple lines, 
firstly you need to extract the line level images from it and predict each image by the model. '''