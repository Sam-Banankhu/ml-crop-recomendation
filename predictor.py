import pickle
import pandas as pd
import numpy as np
import joblib
import warnings

# Ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Your code that produces warning messages
warnings.filterwarnings("ignore")


class Crop_predictor:
    def __init__(self, N, P, K):
        self.nitrogen = int(N)
        self.phosphorus = int(P)
        self.potassium = (K)
    
    
    def deserialize(self):
        # de-serialize mlp_nn.pkl file into an object called model using pickle
        # with open('model_1.pkl', 'rb') as handle:
        model = joblib.load('model_1.sav')
        # print('model loaded')
        return model
  
    def predict(self):
        model = self.deserialize()
        args = np.array([[self.nitrogen, self.phosphorus, self.potassium]])
        return model.predict(args)




    # N = request.args.get('N')
    # P = request.args.get('P')
    # K = request.args.get('K')