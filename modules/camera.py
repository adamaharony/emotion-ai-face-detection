import cv2
from modules.model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('./static/haar_face.xml')
eyec = cv2.CascadeClassifier('./static/haar_eye.xml')
model = FacialExpressionModel("./static/model.json", "./static/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX  # TODO: REPLACE FONT IN PRODUCTION


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        grey_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(
            grey_fr,  # Forwarding each greyscale frame to the classifier
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(50, 50)  # Minimum size of a face (in pixels)
        )
        eyes = eyec.detectMultiScale(
            grey_fr,  # Forwarding each greyscale frame to the classifier
            scaleFactor=1.2,
            minNeighbors=15,
            minSize=(15, 15)  # Minimum size of an eye (in pixels)
        )

        for (x, y, w, h) in faces:
            # getting the face part to feed into the model
            fc = grey_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            preds = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, f"NOW: {preds[0]}", (x, y - 50), font, 1, (255, 255, 0), 2)
            cv2.putText(fr, f"OVERALL: {preds[1]}", (x, y - 10), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), model.calculate_suspicious(), 2)

        for (x, y, w, h) in eyes:
            cv2.rectangle(fr, (x, y), (x + w, y + h), model.calculate_suspicious(), 1)  # BGR format

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
