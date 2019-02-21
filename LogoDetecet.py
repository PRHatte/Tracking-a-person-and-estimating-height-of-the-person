from scipy.spatial import distance as dist
import cv2
import numpy as np


detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

people_cascade = cv2.CascadeClassifier("C:/Users/prh170230/Downloads/haarcascade_upperbody.xml")
face_cascade = cv2.CascadeClassifier("C:/Users/prh170230/Downloads/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/prh170230/Downloads/haarcascade_eye.xml")
LOGOImg=cv2.imread("C:/Users/prh170230/Downloads/logo.jpg",0)
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
        #cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        x=[np.int32(queryBorder)]
        cv2.rectangle(QueryImgBGR,tuple(x[0][0][0]),tuple(x[0][0][2]),(0,0,255),3)
        a=x[0][0][0][0]
        b=x[0][0][0][1]
        c=x[0][0][2][0]
        d=x[0][0][2][1]
        mid=int((a+c)/2)
        wid1=int((3*a-mid)/2)
        wid2=int(2*mid-wid1)
        u=(x[0][0][1][1]*1.4-x[0][0][0][1])/0.4
        #cv2.rectangle(QueryImgBGR,(a,2*b-d),(c,int(u)),(255,0,0),3)
        cv2.rectangle(QueryImgBGR,(wid1,2*b-d),(wid2,int(u)),(255,0,0),3)
        #window = QueryImgBGR[2*b-d:int(u),wid1:wid2]
        face= face_cascade.detectMultiScale(QueryImgBGR,1.3,5)
        eyes= eye_cascade.detectMultiScale(QueryImgBGR,1.3,5)
        people=people_cascade.detectMultiScale(QueryImgBGR,1.3,5)
        for(p,q,r,s) in people:
            #cv2.rectangle(QueryImgBGR,(p,q),(p+r,q+s),(0,0,255),2)
            if c>((2*p+r)/2)>a:
                cv2.rectangle(QueryImgBGR,(p,q),(p+r,q+s),(0,0,255),2)
        for (x,y,w,h) in face:
            if(x > wid1 and x+w < wid2):
                cv2.rectangle(QueryImgBGR,(x,y),(x+w,y+h),(0,255,0),2)
        for (x,y,w,h) in eyes:
            if(x > wid1 and x+w < wid2):
                cv2.rectangle(QueryImgBGR,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

