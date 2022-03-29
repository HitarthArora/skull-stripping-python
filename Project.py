import numpy as np
import cv2
from skimage.measure import label 

#import image
image=cv2.imread('brain1.jpg',0)
cv2.imshow("Input", image)
original=image.copy()

#step-1
step1 = cv2.medianBlur(image, 3)

#step-2
def computeMean(image):
    m, n = image.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            sum += image[i][j]
    mean = sum/(m*n)
    return mean

Ti = computeMean(image)

#step-3
def findExtremes(image, Ti):
    m, n = image.shape
    top = m+1
    left = n+1
    bottom = -1
    right = -1
    for i in range(m):
        for j in range(n):
            if image[i][j] > Ti:
                top = min(top, i)
                bottom = max(bottom, i)

    for j in range(n):
        for i in range(m):
            if image[i][j] > Ti:
                left = min(left, j)
                right = max(right, j)
    return top, bottom, left, right


top, bottom, left, right = findExtremes(image, Ti)

#step-4 and step-5
def computeMeanRange(image,top,bottom,left,right):
    m, n = image.shape
    sum = 0
    for i in range(top,bottom+1):
        for j in range(left,right+1):
            sum += image[i][j]
    mean = sum/((right-left+1)*(bottom-top+1))
    return mean

Tf=computeMeanRange(image, top, bottom, left, right)

#step-6 & step-7
def computeMeanMembrane(image,top,bottom,left,right,Tf):
    m, n = image.shape
    sum = 0
    for i in range(top,bottom+1):
        for j in range(left,right+1):
            if(image[i][j]<Tf):
                sum += image[i][j]
    mean = sum/((right-left+1)*(bottom-top+1))
    return mean

T=computeMeanMembrane(image, top, bottom, left, right, Tf)

#step-8
def binaryThresholding(image,T):
    m,n=image.shape
    res=image.copy()
    for i in range(m):
        for j in range(n):
            if image[i][j]>T:
                res[i][j]=255
            else:
                res[i][j]=0
    return res

binaryImage=binaryThresholding(image, T)

#step-9
def opening(image,kernelSize=(13,13)):
    res=image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernelSize)
    out = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    return out

openingImage=opening(binaryImage)

#step-10
def getLargestCC(img):
    image=img.copy()
    image=image/255
    labels = label(image)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC*255

res1=getLargestCC(openingImage)
res1=res1.astype(np.uint8)

#step-11
def closing(image,kernelSize=(21,21)):
    res=image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernelSize)
    output = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    return output

FinalBinaryOutput=closing(res1)

def postProcessing(ori,img):
    m,n=ori.shape
    res=img.copy() 
    for i in range(m):
        for j in range(n):
            if res[i][j]==255:
                res[i][j]=ori[i][j]
    return res

FinalOutput=postProcessing(original,FinalBinaryOutput)
cv2.imshow("FinalOutput", FinalOutput)
cv2.imwrite("FinalOutput1.jpg", FinalOutput)
cv2.waitKey(0)