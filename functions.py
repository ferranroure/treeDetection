
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from fastai.vision import open_image


def sliding_windowMosaicMask(image,mask, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],mask[y:y + windowSize[1], x:x + windowSize[0]])

# receive an image, a patch size and a step, a classifier and a label name,
# create a mask of all the patches that belong to the label according to the classifier
def classifyPatches(mosaic,step,patchSize,classifierPatchSize,learner=None,label=""):
    # create empty output mask
    # create a mask of ones to be added each time a patch is found to contain the label
    outputMask = np.zeros((mosaic.shape[0],mosaic.shape[1]), dtype = "uint8")

    for (x, y, mosaicW,maskW) in sliding_windowMosaicMask(mosaic, outputMask, stepSize=step, windowSize=(patchSize, patchSize)):

        # create a mask of ones to be added each time a patch is found to contain the label
        maskOfOnes = 1*np.ones((maskW.shape[0],maskW.shape[1]), dtype = "uint8")

        # Can we do the following shit in a less ugly way???? (no, learner.predict(mosaicW) does not work)
        tempImage = cv2.resize(mosaicW,(classifierPatchSize,classifierPatchSize))
        cv2.imwrite("./tempImage.jpg",tempImage)
        img=open_image("./tempImage.jpg")
        pred_class,pred_idx,outputs =learner.predict(img)
        if label in str(pred_class).split(";"): maskW=np.add(maskW,maskOfOnes)

    # At the end of the loop, we have added ones every time a patch has been classified as belonging to the class
    oMaskMax=np.max(ouputMask)
    heatMapPerc=0.5
    ouputMask[ouputMask<int(oMaskMax*heatMapPerc)]=0
    ouputMask[ouputMask!=0]=255
    print("output mask max!! "+str(oMaskMax))
    return outputMask

def show_img(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def borderPoint(image,point):
	margin=100
	top1=image.shape[0]
	top2=image.shape[1]

	return point[0]<margin or (top1-point[0])<margin or point[1]<margin or (top2-point[1])<margin

# Function to take a binary image and output the center of masses of its connected regions
# THIS METHOD IS A COPY OF crownSectmenterEvaluator method! must be deleted!!!
def listFromBinary(fileName):
	#open filename
	im=cv2.imread(fileName,cv2.IMREAD_GRAYSCALE)
	if im is None: return []
	else:
		mask = cv2.threshold(255-im, 40, 255, cv2.THRESH_BINARY)[1]

		#compute connected components
		numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)
		#print("crownSegmenterEvaluator, found "+str(numLabels)+" "+str(len(centroids))+" points for file "+fileName)

		#im2 = 255 * np. ones(shape=[im.shape[0], im.shape[1], 1], dtype=np. uint8)

		#print(" listFromBinary, found  "+str(len(centroids)))
		#print(centroids)

		newCentroids=[]
		for c in centroids:
			if not borderPoint(im,c):newCentroids.append(c)
		#print(" listFromBinary, refined  "+str(len(newCentroids)))
		#print(newCentroids)

		return newCentroids[1:]


def getSquare(w_size, p, img):


	height, width, channels = img.shape

	isInside = (int(p[0])-w_size//2) >= 0 and (int(p[0])+w_size//2) < width and (int(p[1])-w_size//2) >= 0 and (int(p[1])+w_size//2) < height

	assert isInside, "The required window is out of bounds of the input image"

	return img[int(p[0])-w_size//2:int(p[0])+w_size//2, int(p[1])-w_size//2:int(p[1])+w_size//2]

def isInLayer(center,layer):
	return layer[int(center[0]),int(center[1])]==255
