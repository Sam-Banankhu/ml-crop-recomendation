import pickle
import pandas as pd
import numpy as np
import joblib
import warnings
# Your code that produces warning messages
warnings.filterwarnings("ignore")
# Ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

class CropPredictor:
    def __init__(self, N, P, K):
        self.nitrogen = int(N)
        self.phosphorus = int(P)
        self.potassium = (K)
    
    def deserialize(self):
        # de-serialize model_1.sav file into an object called model using joblib
        model = joblib.load('model_1.sav')
        cols = joblib.load('features_1.sav')
        return model, cols
  
    def predict(self):
        model, cols = self.deserialize()
        args = np.array([[int(self.nitrogen), int(self.phosphorus), int(self.potassium)]])
        df = pd.DataFrame(data = args, columns=cols)
        # print(df)
        return model.predict(df)




    # N = request.args.get('N')
    # P = request.args.get('P')
    # K = request.args.get('K')