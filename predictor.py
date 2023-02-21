import pickle
import pandas as pd
import numpy as np
import joblib
import warnings

# Ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Your code that produces warning messages
warnings.filterwarnings("ignore")


class Crop_predictor():
    def __init__(self):
        pass
    
    
    def deserialize(self):
        # de-serialize mlp_nn.pkl file into an object called model using pickle
        # with open('model_1.pkl', 'rb') as handle:
        model = joblib.load('model_1.sav')
        # print('model loaded')
        return model
  
    def predict(self,  N,  P, K):
        model = self.deserialize()
        return model.predict(np.array([[N, P, K]]))




    # N = request.args.get('N')
    # P = request.args.get('P')
    # K = request.args.get('K')