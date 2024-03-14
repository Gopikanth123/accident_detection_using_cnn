import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import matplotlib.pyplot as plt

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def start_application():
    video = cv2.VideoCapture('cars.mp4') # for camera use video = cv2.VideoCapture(0)
    fig, ax = plt.subplots()
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob = (round(prob[0][0]*100, 2))
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.pause(0.01)

    video.release()

if __name__ == '__main__':
    start_application()
