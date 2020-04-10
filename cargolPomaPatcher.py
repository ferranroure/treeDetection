# This method takes the point coordinates of each tree top given by crownSegementerEvaluator and extracts
# a squared patch arround each one. Then, uses the classified masks of the mosaics to know in which species belongs.
# Then stores each small labeled patch in a folder.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from functions import *

mosaicInfo = namedtuple("mosaicInfo","path mosaicFile numClasses layerNameList layerFileList outputFolder " )



def interpretParameters(paramFile,verbose=False):
	# read the parameter file line by line
	f = open(paramFile, "r")
	patchSize=-1
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
			layerNameList=[]
			layerFileList=[]

			# read the number of layers and set up reading loop
			filePath=lineList[1]
			mosaic=lineList[2]
			numClasses=int(lineList[3])
			outputFolder=lineList[4+numClasses*2].strip()
			for i in range(4,numClasses*2+3,2):
				layerNameList.append(lineList[i])
				layerFileList.append(filePath+lineList[i+1])

			#make dictionary entry for this mosaic
			mosaicDict[mosaic]=mosaicInfo(filePath,mosaic,numClasses,layerNameList,layerFileList,outputFolder)
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



def main(argv):
	try:
		# verbose = False
		patchSize, mosaicDict = interpretParameters(argv[1])

		#if verbose: print(mosaicDict)
		for mosaicName, mosaicInfo in mosaicDict.items(): # FOR EACH MOSAIC

			mosaicFile = mosaicInfo.path + mosaicInfo.mosaicFile
			outputFolder = mosaicInfo.path + mosaicInfo.outputFolder + "/"
			mosaic = cv2.imread(mosaicFile, cv2.IMREAD_COLOR)


			for layerFileName in mosaicInfo.layerFileList: # FOR EACH CLASS 

				centroids = listFromBinary(layerFileName)
				counter = 0

				layers=[cv2.bitwise_not(cv2.imread(layerFileName, cv2.IMREAD_GRAYSCALE)) for layerFileName in mosaicInfo.layerFileList]

				for cent in centroids: # FOR EACH ELEMENT

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
