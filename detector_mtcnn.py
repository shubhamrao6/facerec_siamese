import os
import numpy as np
import cv2
from model import *
from mtcnn.mtcnn import MTCNN

img_db = np.load('image_database.npy')
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')

def recognizer(img, img_db):

	img = cv2.resize(img, (46,56))
	n, w, h = img_db.shape
	values = []

	for i in range(n//10):
		value = 0
		r = np.random.randint(i*10, i*10+10, 3)
		for j in r:
#            values.append(model.predict([img_db[j].reshape(1,1,56,46)/255, img.reshape(1,1,56,46)/255]))
			value += model.predict([img_db[j].reshape(1,1,56,46)/255, img.reshape(1,1,56,46)/255])
		values.append(value/4)
        
	return np.argmin(values), np.min(values)

def detect(gray, frame):
#    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    roi_gray = np.random.rand(100,100)
    a = 0
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        if recognizer(roi_gray, img_db)[1] > 0.2:
            cv2.putText(frame,'unknown',(x,y),cv2.FONT_HERSHEY_COMPLEX,.5,(0,0,0),1)    
        elif recognizer(roi_gray, img_db)[0] == 0:
            cv2.putText(frame,'Shubham',(x,y),cv2.FONT_HERSHEY_COMPLEX,.5,(0,0,0),1)
        elif recognizer(roi_gray, img_db)[0] == 1:
            cv2.putText(frame,'Jhalak',(x,y),cv2.FONT_HERSHEY_COMPLEX,.5,(0,0,0),1)
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        a += 1
    return frame

if __name__ == "__main__":
	video_capture = cv2.VideoCapture(0)
	# fourcc = cv2.VideoWriter_fourcc(*'XVID') 
	# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 
	while True:
		_, frame = video_capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		canvas = detect(gray, frame)
		cv2.imshow('Video', canvas)
		if cv2.waitKey(1) & 0xFF == ord('q'):
	    		break
	#     out.write(canvas)

	video_capture.release()
	# out.release()  
	cv2.destroyAllWindows()
