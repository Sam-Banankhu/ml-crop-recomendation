import json
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
from predictor import crop_predictor

app = Flask(__name__)
CORS(app)
@app.route("/crop/",methods=['GET'])
def return_price():
  date = request.args.get('N')
  month = request.args.get('P')
  year = request.args.get('K')
  crop = crop_predictor.predict(date, month, year) 
  price_dict = {
                'model':'model',
                'crop': crop,
                }
  return jsonify(price_dict)

@app.route("/",methods=['GET'])
def default():
  return "<h1> Welcome to crop recommendation system <h1>"

if __name__ == "__main__":
    app.run()