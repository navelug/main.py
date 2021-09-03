from silence_tensorflow import silence_tensorflow; silence_tensorflow()
from keras.models import load_model
import cv2
from sklearn.preprocessing import minmax_scale
import tkinter as tk
import numpy
import subprocess
import json

model = load_model('model_NN_20_03_21.h5') # Load trained model

cam = cv2.VideoCapture(0) # Open webcam
cv2.namedWindow("Hand Pose Classification")
while True:
    ret, frame = cam.read()
    cv2.putText(frame, 'Preform sign & press SPACE key...', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 241, 93), 2, cv2.LINE_4)
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k % 256 == 27: # ESC pressed - quit program
        break
    elif k % 256 == 32: # SPACE pressed - capture image
        img_name = "C:\\OpenPose\\test\\opencv_frame.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        subprocess.call([r'C:\OpenPose\test\cmd.bat']) # Render image
        with open('C:\\OpenPose\\test\\opencv_frame_keypoints.json') as f: # Load processed data
            data = json.load(f)
        with open('C:\\OpenPose\\test\\opencv_frame_keypoints_out.npy', 'wb') as g: # Extract hand keypoints
            numpy.save(g, numpy.array(data['people'][0]['hand_left_keypoints_2d']))
        print("{} rendered!".format(img_name))
        data = minmax_scale(numpy.load('C:\\OpenPose\\test\\opencv_frame_keypoints_out.npy'))
        text = "Prediction :" + str(int(numpy.argmax(model.predict(data.reshape(1, 63)), axis=1))) # Model predict
        while True:
            ret, frame = cam.read()
            cv2.putText(frame, text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 241, 93), 2, cv2.LINE_4) # Print result
            cv2.putText(frame, "hit SPACE to restart", (275, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 241, 93), 2, cv2.LINE_4)
            cv2.imshow("test", frame)
            k = cv2.waitKey(1)
            if k % 256 == 32: # SPACE pressed - next round
                break
cam.release()
cv2.destroyAllWindows()