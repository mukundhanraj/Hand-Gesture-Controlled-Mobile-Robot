import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import fftpack
import imutils
from pprint import pprint
import math
import io
import socket
import struct
import time
import pickle
import zlib

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.241.5', 50000))
connection = client_socket.makefile('wb')

vid = cv2.VideoCapture(0)
vid.set(3, 320);
vid.set(4, 240);
vid.isOpened()
img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


def Fourier(gray):
    
        y1 = fftpack.fft2(gray)
        
        y2 = fftpack.fftshift(y1)
        
        (w, h) = gray.shape
        half_w, half_h = int(w/2), int(h/2)
        
        # high pass filter
        n = 3
        y2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0

        y3 = fftpack.ifftshift(y2)
        
        y4 = fftpack.ifft2(y3)
        
        y = np.uint8(np.abs(y4))
        
        return y
# Finding the corners of the tag from world co-ordinates
def contours(y):
    
    edged = cv2.Canny(y, 50, 150)
   
    ret, thresh = cv2.threshold(edged, 127, 255, 0)
    
    contours= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    
    
    contours = sorted(contours[1], key = cv2.contourArea, reverse = True)[:5]
    
    #Finding the corners of the tag
    for cnt in contours:
        
        perimeter = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.09*perimeter,True)
        area = cv2.contourArea(cnt)
        print(area)
        if cv2.contourArea(cnt) < 7000 and cv2.contourArea(cnt) > 1500 and len(approx) == 4:
            
            apx = approx
            apx = apx.reshape(4,2)
            
            corn = np.zeros((4,2))
            
            add_cnt = apx.sum(axis=1)
            
            corn[0] = apx[np.argmin(add_cnt)]
            corn[2] = apx[np.argmax(add_cnt)]
            
            diff_cnt = np.diff(apx,axis=1)
            
            corn[1] = apx[np.argmin(diff_cnt)]
            corn[3] = apx[np.argmax(diff_cnt)]
           
            return corn
    




def homography(cornlist,wlist):
    Alist = []
    for i in range(len(cornlist)):
        u, v = cornlist[i][0],cornlist[i][1]
        X, Y = wlist[i][0],wlist[i][1]
        Alist.append([X , Y , 1 , 0 , 0 , 0 , - X * u , - Y * u , - u])
        Alist.append([0 , 0 , 0 , X , Y , 1 , - X * v , - Y * v , - v]) 
    A = np.array(Alist)
  
    U, sigma, VT = np.linalg.svd(A)
    
    v= VT.T
    
    rv = v[:,8]/v[8][8]
    rv = rv.reshape((3,3))
    
    return rv

# Reorienting the Tag
def reorient():
    
    atag = [[0,0],[0,500],[500,500],[500,0]]
    newatag = np.zeros((500,500,3))
    #sh = newatag.shape
    #A = np.zeros(shape=(8,9))
    Alist = []
    for i in range(4):
        u, v = atag[i][0],atag[i][1]
        X, Y = corn[i][0],corn[i][1]
        Alist.append([X , Y , 1 , 0 , 0 , 0 , - X * u , - Y * u , - u])
        Alist.append([0 , 0 , 0 , X , Y , 1 , - X * v , - Y * v , - v]) 
        A = np.array(Alist)
    U, sigma, VT = np.linalg.svd(A)
    
    v= VT.T
    
    rv = v[:,8]/v[8][8]
    rv = rv.reshape((3,3))
    
    #r = homography(atag,reclist)
    
    rv_inv = np.linalg.inv(rv)
    
    for i in range(500):
        for j in range(500):
            wcoors=np.array([i,j,1])
            C = np.dot(rv_inv,wcoors)
            C = C/C[2]
            if (320 > C[0] > 0) and (240 > C[1] > 0):
                newatag[i][j] = input_img[int(C[1])][int(C[0])]
    
    newatag = newatag.astype(np.uint8)
    
    return newatag
    

while True:
    
   
    ret, frame = vid.read()
    
    if ret == True:
        
        
        input_img = cv2.resize(frame,(320,240)) 
        b1 = input_img.copy()   
        b2 = input_img.copy()
        b3 = input_img.copy() 
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) 
        
        
        
        corn = contours(gray)
        
        if corn is not None:
          
            cornlist = corn.tolist()
            cornlist2 = corn.tolist().copy()
            wlist = list()
            
            newatag = reorient()
            if newatag is not None:
              
                  
                    wlist = [[0,0],[0,500],[500,500],[500,0]]
                    
                    rv1 = homography(cornlist,wlist)
                
                    img2 = cv2.imread('blimp.png')
                    img2 = cv2.resize(img2,(500,500))
                    
                    test = img2.shape
                    
                    new_img = np.zeros((test[0],test[1]))
                    
                    new_coor = []
                    
                    for i in range(test[0]):
                        for j in range(test[1]):
                            coor = np.array([[i],[j],[1]])
                            q = np.dot(rv1,coor)
                            new_x = q[0]/q[2]
                            new_y = q[1]/q[2]
                            pixel = img2[i][j]
                            new_coor.append([int(new_x),int(new_y),pixel])
                    
                    for i in range(len(new_coor)):
                        if (320 > new_coor[i][0] > 0) and (240 > new_coor[i][1] > 0):
                                input_img[new_coor[i][1]][new_coor[i][0]] = new_coor[i][2]
                    
                    #cv2.imshow('blimp_live',input_img)
                    result, frame = cv2.imencode('.jpg', input_img, encode_param)
                #   data = zlib.compress(pickle.dumps(frame, 0))
                    data = pickle.dumps(frame, 0)
                    size = len(data)


                    print("{}: {}".format(img_counter, size))
                    client_socket.sendall(struct.pack(">L", size) + data)
                    img_counter += 1
                    continue        
        
        result, frame = cv2.imencode('.jpg', input_img, encode_param)
        #data = zlib.compress(pickle.dumps(frame, 0))
        data = pickle.dumps(frame, 0)
        size = len(data)


        print("{}: {}".format(img_counter, size))
        client_socket.sendall(struct.pack(">L", size) + data)
        img_counter += 1

        
    else:
        break
vid.release()    
cv2.destroyAllWindows() 
 

