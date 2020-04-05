import numpy as np
import matplotlib.pyplot as pyplot
import cv2

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')



video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

n = 1
while True:
    
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cv2.imshow('Video', gray)
    
    key=cv2.waitKey(30)
    
    faces = np.array(faces)
    
    if key==ord('q') or n==11:
        break
    
    if (key==ord('c')) & (faces.shape==(1,4)):
        faces = np.array(faces).reshape((4,1)) 
        face = gray[int(faces[1]):int(faces[1])+int(faces[3]), int(faces[0]):int(faces[0])+int(faces[2])]
        cv2.imshow("imshow2",face)
        #cv2.imwrite('C:/Users/hp/Desktop/facerec/Siamese/image'+str(np.random.randint(0, 100))+'.pgm', face,params=(cv2.IMWRITE_PXM_BINARY,0))
        cv2.imwrite('C:/Users/hp/Desktop/facerec/Siamese/face/'+str(n)+'.pgm', 
                    face,params=(cv2.IMWRITE_PXM_BINARY,0))
        print("Wrote Image")
        n += 1
    
video_capture.release()
cv2.destroyAllWindows()

