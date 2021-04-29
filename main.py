import cv2
from matplotlib import pyplot as pyplot
import numpy as np
import math as math

#Divides the image index as split_size*split_size split size must have pow of a 2
def Divide_Img(img,split_size):
    sliced=np.split(img,split_size,axis=0)
    return [np.split(img_slice,split_size,axis=1) for img_slice in sliced]

#These two functions are just trick to reduce time complexity
#While they are reducing time complexity of Threshold and FindTuner Functions they are increasing space complexity
#CreateImageIndex should be given as last parameter to Threshold function
#CreateBlockIndex using in Binarize function
def CreateBlockIndex(split_size):
    index=[]
    for x in range(split_size):
        for y in range(split_size):
            index.append((x,y))
    return index
def CreateImageIndex(img):
    imgsize= img.shape
    index=[]
    for x in range(imgsize[0]):
        for y in range(imgsize[1]):
            index.append((x,y))
    return index

#Threshold function to binarize image as globally      
def Threshold(p_img,value,max,inv,index):
    min=0
    if inv:
        min=max
        max=0
    for i in index:
        if p_img[i[0]][i[1]]>=value:
                p_img[i[0]][i[1]]=max
        else:
            p_img[i[0]][i[1]]=min
    return p_img

#Function to find tuner value to use in Binarize funciton it helps to reduce noise and 
def FindTuner(img,split_size):
    tuner = 0
    var = np.var(img)
    log=math.log2(split_size)
    if(var<log):
        tuner=log-math.floor(var)
    return tuner

#It resize the image as 1024x1024, divides it, binarizes each image locally and merge them back
#If use tuner function given False tuner value not be used and resluts may be worse
def Binarize(img,split_size=128,invert=False,use_tuner=True):
    img = cv2.resize(img,(512,512))
    blocks = Divide_Img(img,split_size)
    index = CreateBlockIndex(split_size)
    img_index=CreateImageIndex(blocks[0][0])
    count=0
    for i in index:
        tuner=0
        if (use_tuner):
            tuner=FindTuner(blocks[i[0]][i[1]],split_size)
        blocks[i[0]][i[1]]=Threshold(blocks[i[0]][i[1]],np.mean(blocks[i[0]][i[1]])-tuner,255,invert,img_index)
    return np.block(blocks)

if __name__=='__main__':
    #Read and show original image as grayscale
    img = cv2.imread('test/yuz_org.jpg',0)
    org_size=img.shape
    cv2.imshow("Original",img)
    cv2.waitKey(0)

    #Global Threshold using our own threshold function and show
    glob = Threshold(img.copy(),127,255,False,CreateImageIndex(img))
    cv2.imshow("Global 127",glob)
    cv2.waitKey(0)

    # Otsu's thresholding using opencv
    ret2,otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("Otsu",otsu)
    cv2.waitKey(0)

    #Local Threshold using our own function
    local=Binarize(img.copy(),256,False,True)
    #Resize it back to original sizes
    local = cv2.resize(local,(org_size[1],org_size[0]))
    cv2.imshow("Local:",local)
    cv2.waitKey(0)