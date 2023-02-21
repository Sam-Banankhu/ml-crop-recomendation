import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from predictor import Crop_predictor

app = Flask(__name__)
CORS(app)

@app.route("/crop", methods=['POST'])
def return_crop():
    if request.content_type == 'application/json':
        npk = request.get_json()
        N = npk['N']
        P = npk['P']
        K = npk['K']

    elif request.content_type == 'application/x-www-form-urlencoded':
        # npk = request.form.to_dict()
        N = request.form['N']
        P = request.form['P']
        K = request.form['K']
    else:
        return jsonify({'error': 'Unsupported Content-Type'}), 400

    model = Crop_predictor(N=N,P=P,K=K)

    crop = model.predict()
    crop_dict = {
        'model': 'model',
        'crop': crop,
    }

    return jsonify(crop_dict)

@app.route("/", methods=['POST', 'GET'])
def default():
    return "<h1>Welcome to crop recommendation system</h1>"

if __name__ == "__main__":
    app.run(debug=True)
