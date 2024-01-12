import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# 初始化音频播放器
mixer.init()
# 加载音频文件
sound = mixer.Sound('alarm.wav')

# 加载人脸、左眼和右眼的Haar cascade分类器
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('models/cnncat2.h5')# 加载训练好的CNN模型
path = os.getcwd()
cap = cv2.VideoCapture(0) # 捕获对象
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

# 无限循环，捕获摄像头的每一帧
while(True):
    ret, frame = cap.read() # 可以读取每一帧
    height,width = frame.shape[:2]
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 分别使用相应Haar cascade分类器逐一检测人脸，左眼，右眼
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    # 用以警示的矩形框
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    # 人脸矩形框
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    # 右眼
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w] #提取右眼区域
        count=count+1 # 帧计数+1
        # 转换为灰度图，调整像素，归一化
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1) # 重塑形状
        r_eye = np.expand_dims(r_eye,axis=0) # 增加维度
        # 使用模型预测右眼状态
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    # 左眼
    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    # 根据左右眼是否都闭合，计算score
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    if(score>15):
        #驾驶员打瞌睡了，因此使用警报声对其进行提醒
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play() # 播放警报
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2 # 增加画框线条的粗细
        else:
            thicc=thicc-2 # 减小画框线条的粗细
            if(thicc<2):
                thicc=2
        # 红色矩形框做警示
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() # 释放摄像头
cv2.destroyAllWindows() # 关闭所有窗口
