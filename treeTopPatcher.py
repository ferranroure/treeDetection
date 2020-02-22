# This method takes the point coordinates of each tree top given by crownSegementerEvaluator and extracts
# a squared patch arround each one. Then, uses the classified masks of the mosaics to know in which species belongs.
# Then stores each small labeled patch in a folder.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

mosaicInfo = namedtuple("mosaicInfo","path mosaicFile mosaicTopsFile numClasses layerNameList layerFileList outputFolder " )

def interpretParameters(paramFile,verbose=False):
	# read the parameter file line by line
	f = open(paramFile, "r")
	patchSize=-1
	layerNameList=[]
	layerFileList=[]
	mosaicDict={}

	for x in f:
		lineList=x.split(" ")
		# read every line
		first=lineList[0]

		if first[0]=="#": #if the first character is # treat as a comment
			if verbose:print("COMMENT: "+str(lineList))
		elif first=="\n":# account for blank lines, do nothing
			pass
		elif first=="patchSize":
			patchSize=int(lineList[1].strip())
			if verbose:print("Read Patch Size : "+str(patchSize))
		elif first=="mosaic":
			# read the number of layers and set up reading loop
			filePath=lineList[1]
			mosaic=lineList[2]
			mosaicTops = lineList[3]
			numClasses=int(lineList[4])
			outputFolder=lineList[5+numClasses*2].strip()
			for i in range(5,numClasses*2+4,2):
				layerNameList.append(lineList[i])
				layerFileList.append(filePath+lineList[i+1])

			#make dictionary entry for this mosaic
			mosaicDict[mosaic]=mosaicInfo(filePath,mosaic,mosaicTops,numClasses,layerNameList,layerFileList,outputFolder)
			if verbose:
				print("\n\n\n")
				print(mosaicDict[mosaic])
				print("\n\n\n")
				#print("Read layers and file : ")
				#print("filePath "+filePath)
				#print("mosaic "+mosaic)
				#print("num Classes "+str(numClasses))
				#print("layerName List "+str(layerNameList))
				#print("layer List "+str(layerList))
				#print("outputFolder "+outputFolder)
		else:
			raise Exception("ImagePatchAnnotator:interpretParameters, reading parameters, received wrong parameter "+str(lineList))

		if verbose:(print(mosaicDict))

	return patchSize,mosaicDict


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


def main(argv):
	try:
		# verbose = False
		patchSize, mosaicDict = interpretParameters(argv[1])

		#if verbose: print(mosaicDict)
		for mosaicName, mosaicInfo in mosaicDict.items():

			mosaicTopsFile = mosaicInfo.path + mosaicInfo.mosaicTopsFile
			mosaicFile = mosaicInfo.path + mosaicInfo.mosaicFile
			outputFolder = mosaicInfo.path + mosaicInfo.outputFolder + "/"

			# if verbose: print("\n\nstarting processing of first mosaic and layers "+str(v)+"\n\n")
			treetops_mask = cv2.imread(mosaicTopsFile, cv2.IMREAD_GRAYSCALE)

			mosaic = cv2.imread(mosaicFile, cv2.IMREAD_COLOR)

			centroids = listFromBinary(mosaicTopsFile)
			counter = 0

			layers=[255-cv2.imread(layerFileName, cv2.IMREAD_GRAYSCALE) for layerFileName in mosaicInfo.layerFileList]

			for cent in centroids:

				try:
					# opencv works with inverted coords, so we have to invert ours.
					square = getSquare(patchSize, (cent[1],cent[0]), mosaic)
					className="EMPTYCLASS"
					for i in range(mosaicInfo.numClasses):
						#print(str((cent[1],cent[0]))+" TO BE CHECKED FOR CLASS "+mosaicInfo.layerNameList[i])

						if isInLayer((cent[1],cent[0]),layers[i]):
							if className!="EMPTYCLASS":
								raise Exception(str((cent[1],cent[0]))+"center belongs to two classes,  "+className+" and "+mosaicInfo.layerNameList[i])
							#print("found that "+str((cent[1],cent[0]))+" belongs to "+mosaicInfo.layerNameList[i])
							className=mosaicInfo.layerNameList[i]

					if className=="EMPTYCLASS":
						#raise Exception(str((cent[1],cent[0]))+"center belongs to no class")
						print(str((cent[1],cent[0]))+"center belongs to no class")
					else:
						cv2.imwrite(outputFolder+"SP"+className+"PATCH"+str(counter)+".jpg", square)
					counter+=1

				except AssertionError as error:
					print(error)


	except AssertionError as error:

		print(error)


# Exectuion example -> python treeTopPatcher.py <path_to_params_file>

if __name__ == "__main__":
	main(sys.argv)
