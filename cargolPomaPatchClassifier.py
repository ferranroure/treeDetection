import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
import numpy as np
import sys
from fastai import *
from fastai.vision import *
from functions import *


def setRandomSeed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
#Remember to use num_workers=0 when creating the DataBunch.

def main(argv):

    #Before we begin, check installation
    print("Checking if cuda is available "+str(torch.cuda.is_available()))
    print("Checking if cuddn is available "+str(torch.backends.cudnn.enabled))

    #Parameters
    PATH = pathlib.Path(argv[1])

    #print("Data path in "+str(PATH))
    # To see what files are in the PATH folder
    #print(PATH.ls())

    # argv[2] contains the model file name, where the weights of the network are stored
    # if we are training, it will be the file that we will use to SAVE the model
    # if we are not training, we will LOAD a model with this name, must match the architecture
    modelFileName=argv[2]

    # Argv[3] Define architecture, tested resnet and vgg, there are more
    if argv[3]=="res": arch=models.resnet50
    elif argv[3]=="vgg":arch=models.vgg16_bn
    elif argv[3]=="squ":arch=models.squeezenet1_0
    elif argv[3]=="dense":arch=models.densenet121
    elif argv[3]=="alex":arch=models.alexnet

    else: raise Exception("fastai cargol poma, unrecognised architecture")
    # many more models are possible
    #https://docs.fast.ai/vision.models.html

    # argv[4] contains a code on what to do, if it contains "t" it will train
    # if it contains "p" it will predict categories for new images
    # Are we training the model or just loading one?
    training="t" in argv[4]

    # are we going to predict images not in the training/testing set?
    predict="p" in argv[4]

    offset=0
    if training:
        numEpochs=int(argv[5])
        lr=float(argv[6])
        offset=2
        print("TESTING ARCHITECTURE "+argv[3]+" with learning rate "+str(lr))
    if predict:
        #the following parameter (either 5 or 7 depending on whether training is True)
        # contains a file with an image to be traversed to predict where the apple snail is
        predictFileName=argv[5+offset]

    bs=32 # size of batch, if you run out of memory switch to 16 (slower)

    # Get data file names, each image name will be extracted with REGULAR expressions
    fnames = get_image_files(PATH)
    pat = r'/SP([^/]+)PATCH\d+.jpg$'
    # Regular expression for file name!!! Some references in case you are interested
    #https://docs.python.org/3/howto/regex.html
    #https://www.guru99.com/python-regular-expressions-complete-tutorial.html
    randomSeed=42
    setRandomSeed(randomSeed)

    # Read data from the path and using the regular expression defined in "pat"
    data = ImageDataBunch.from_name_re(PATH, fnames, pat,
    ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
    print("Printing classes found! ")
    print(data.classes)

    learn = cnn_learner(data, arch, metrics=error_rate)

    setRandomSeed(randomSeed)

    if training:
        learn.fit_one_cycle(numEpochs, slice(lr))
        learn.save(modelFileName, return_path=True)
        interp = ClassificationInterpretation.from_learner(learn)
        conf=interp.confusion_matrix()
        print(conf)
        #valid=data.valid_ds.x.items[:]
        #print("validation data "+str(valid))
        # Make confusion matrix out of these!!!!
#        for x in valid:
#            aux=x.split("/")[-1].split(".")[0]
            # now copy the correct labels on to the validation dictionary
#            self.validFileDict[aux]=self.fileDict[aux]
            #print(self.validFileDict[aux])

    else: # Not training, load model
        learn.load(modelFileName)

        #now   predictFileName is a "mosaic where we will call function "classifyPatches", should be something along the lines of
        # these parameters should not be entered by code
        patchSize=50
        step=50 # this should be <= patchSize, if ==, non-overlapping patches
        classifierPatchSize=100

        classifyPatches(predictFileName,step,patchSize,classifierPatchSize,learner=learn,label="")

#    print("model trained or loaded, now compute predictions? "+str(predict))
#    if(predict):
#        f=open(predictFileName, "r")
#        f1=f.readlines()
#        for line in f1:
#            imageToPredict=line.strip()
#            print("\n\n\npredicting "+imageToPredict)
#            # As loading a model takes time, best to have a list of files here
#            img = open_image(imageToPredict)
#            pred_class,pred_idx,outputs =learn.predict(img)
#            print("Predicted class was "+str(pred_class))
#            print("This class had index "+str(pred_idx.item()))
#            print("The predicted class had probability "+str(outputs[pred_idx].item()))
            #print("probabilities to belong to each of the classes"+str(outputs))


if __name__ == '__main__':
    main(sys.argv)
