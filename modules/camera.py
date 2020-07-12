import cv2
from modules.model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('./static/haar_face.xml')
model = FacialExpressionModel("./static/model.json", "./static/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX     # TODO: REPLACE FONT IN PRODUCTION


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            preds = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, f"NOW: {preds[0]}", (x, y - 50), font, 1, (255, 255, 0), 2)
            cv2.putText(fr, f"OVERALL: {preds[1]}", (x, y - 10), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), model.calculate_suspicious(), 2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
