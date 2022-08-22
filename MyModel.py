import numpy as np
import model_scoring
from model_scoring import load_model, run_scoring
import warnings

warnings.filterwarnings("ignore")

class MyModel(object):
    def __init__(self):
        print("Initializing")


    def health_status(self):
        response = "Service Running"
        return response
    
    # Predict with multiple models
    def predict(self,X,features_names):
        # print(f"Received input {X}")
        score_model_id = str(X[0][0])
        model_input = X[1:]
        # print(f"Model input(s) {model_input}")
        print("********************************************************")
        model_input_array = []
        for i in model_input:
            model_input_array.append(np.array(i))
        model_input_array = np.array(model_input_array)
        array_shape = model_input_array.shape

        if len(array_shape) != 2:
            model_input_array = model_input_array.reshape((array_shape[1],array_shape[2]))
        print(f"Model input(s) {model_input_array}")

        print(f"Predicting with {score_model_id}")
        model = load_model(score_model_id)

        res = run_scoring(model, model_input_array)

        print(f"Prediction: {res}")
        return res
