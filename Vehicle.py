import cv2
import numpy as np

# Web camera
cap = cv2.VideoCapture('Your_Video.mp4') #Example : D:\\Vehicle\\video.mp4

min_width_rect=80 #Minimum width of the rectangle on car
min_height_rect=120 #Minimum height of the rectangle on car


count_line_position=540

#Initialize Substractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect=[]
offset=6  #Allowable error between pixel
counter=0




while True:
    ret,frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    countershape, h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1,(15,count_line_position),(1920,count_line_position+80),(255,127,0),4)

    for (i,c) in enumerate(countershape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter=(w>=min_width_rect) and (w>=min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Vehicle" + str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0))


        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for(x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
            cv2.line(frame1,(15,count_line_position),(1920,count_line_position+80),(0,127,255),3)
            detect.remove((x,y))
            # print("Vehicle Counter : "+str(counter))

    cv2.putText(frame1,"VEHICLE COUNTER : " + str(counter),(450,70),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255))



    cv2.imshow('Video Original',frame1)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()
