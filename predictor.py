import pickle
import pandas as pd
import numpy as np


class crop_predictor():
  def __init__(self):
    pass
  
  def deserialize(self):
    # de-serialize mlp_nn.pkl file into an object called model using pickle
    with open('model.sav', 'rb') as handle:
      model = pickle.load(handle)
      return model
  
  def predict(self, N, P, K):
    model = self.deserialize()
    return model.predict(np.array([[N, P, K]]))