from scipy.spatial import distance as dist
import cv2
import numpy as np


detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

people_cascade = cv2.CascadeClassifier("C:/Users/prh071230/Downloads/haarcascade_upperbody.xml")
face_cascade = cv2.CascadeClassifier("C:/Users/pxh170230/Downloads/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/pxh170230/Downloads/haarcascade_eye.xml")
LOGOImg=cv2.imread("C:/Users/pxb170230/Downloads/book.jpg",0)
trainKP,trainDesc=detector.detectAndCompute(LOGOImg,None)
MIN_MATCH_COUNT=20
cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    for m,n in matches:
        if(m.distance <0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch) >MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=LOGOImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        face = face_cascade.detectMultiScale(QueryImgBGR,1.3,5)
        h_book=queryBorder[0][1][1]-queryBorder[0][0][1]
        for (x,y,w,h) in face:
            cv2.rectangle(QueryImgBGR,(x,y),(x+w,y+h),(255,0,0),2)
            r_f_b=h/h_book
            f_height=r_f_b*22
        
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
r_f_h=15/160
person_height=f_height/r_f_h
print(person_height)
cam.release()
cv2.destroyAllWindows()

