import config
import tensorflow
import tensorflow as tf
import itertools
import numpy as np
from editdistance import eval as edit_distance
from tqdm import tqdm
from data_loader import data_loader
import tensorflow.keras.backend as K

def ctc_custom(args):
    y_pred, labels, input_length, label_length = args
    
    ctc_loss = K.ctc_batch_cost(
        labels, 
        y_pred, 
        input_length, 
        label_length
    )
    p = tensorflow.exp(-ctc_loss)
    gamma = 0.5
    alpha=0.25 
    return alpha*(K.pow((1-p),gamma))*ctc_loss

def load_easter_model(checkpoint_path):
    if checkpoint_path == "Empty":
        checkpoint_path = config.BEST_MODEL_PATH
    try:
        checkpoint = tensorflow.keras.models.load_model(
            checkpoint_path,
            custom_objects={'<lambda>': lambda x, y: y,
            'tensorflow':tf, 'K':K}
        )
        
        EASTER = tensorflow.keras.models.Model(
            checkpoint.get_layer('the_input').input,
            checkpoint.get_layer('Final').output
        )
    except:
        print ("Unable to Load Checkpoint.")
        return None
    return EASTER
    
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
    
def test_on_iam(show = True, partition='test', uncased=False, checkpoint="Empty"):
    
    print ("loading metdata...")
    training_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    validation_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    test_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)

    training_data.trainSet()
    validation_data.validationSet()
    test_data.testSet()
    charlist = training_data.charList
    print ("loading checkpoint...")
    print ("calculating results...")
    
    model = load_easter_model(checkpoint)
    char_error = 0
    total_chars = 0
    
    batches = 1
    while batches > 0:
        batches = batches - 1
        if partition == 'validation':
            print ("Using Validation Partition")
            imgs, truths, _ = validation_data.getValidationImage()
        else:
            print ("Using Test Partition")
            imgs,truths,_ = test_data.getTestImage()

        print ("Number of Samples : ",len(imgs))
        for i in tqdm(range(0,len(imgs))):
            img = imgs[i]
            truth = truths[i].strip(" ").replace("  "," ")
            output = model.predict(img)
            prediction = decoder(output, charlist)
            output = (prediction[0].strip(" ").replace("  ", " "))
            if uncased:
                char_error += edit_distance(output.lower(),truth.lower())
            else:
                char_error += edit_distance(output,truth)
                
            total_chars += len(truth)
            if show:
                print ("Ground Truth :", truth)
                print("Prediction [",edit_distance(output,truth),"]  : ",output)
                print ("*"*50)
    print ("Character error rate is : ",(char_error/total_chars)*100)