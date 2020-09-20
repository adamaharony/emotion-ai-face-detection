from keras.models import model_from_json
import numpy as np


class FacialExpressionModel(object):
    EMOTIONS_LIST = ("Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprised")

    def __init__(self, model_json_file, model_weights_file):
        # loading model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # initialising total & avg expression array and counting times of recognition
        self.runs = 0
        self.tot_exp = np.zeros((1, 7), np.float32)
        self.avg_exp = np.zeros((1, 7), np.float32)

        # loading weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        # after 10 seconds = 300 frames = 300 runs we clear the total evaluation:
        if self.runs % 300 == 0:
            self.runs = 0
            self.tot_exp = 0
        self.tot_exp += self.preds
        self.runs += 1
        self.avg_exp = self.tot_exp / self.runs

        print(
            f"PREDICTION -->  {self.preds}\nTOTAL EXPRESSION -->  {self.tot_exp}\nAVG. EXPRESSION -->  {self.avg_exp}\n\n")
        # determining negative / positive feelings
        self.type_feelings = (self.preds[0, 0] + self.preds[0, 1] + self.preds[0, 2] + self.preds[0, 5]) < 4 / 7
        self.type_feelings_avg = (self.avg_exp[0, 0] + self.avg_exp[0, 1] + self.avg_exp[0, 2] + self.avg_exp[
            0, 5]) < 4 / 7

        return (FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)],
                FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.avg_exp)])

    def calculate_suspicious(self):
        (red, green, blue) = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
        # faking positive:
        if not self.type_feelings_avg:
            return red
        # mostly neutral
        elif self.preds[0, 4] > 1 / 7:
            return blue
        # non problematic
        else:
            return green
