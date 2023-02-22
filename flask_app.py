import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from predictor import CropPredictor

app = Flask(__name__)
CORS(app)

@app.route("/crop", methods=['POST'])
def return_crop():
    if request.content_type == 'application/json':
        # print(request.args.get('N'))
        npk = request.get_json()
        N = npk.get('N')
        P = npk.get('P')
        K = npk.get('K')
    elif request.content_type == 'application/x-www-form-urlencoded':
        # print(request.args.get('N'))
        N = request.args.get('N')
        P = request.args.get('P')
        K = request.args.get('K')
    else:
        return jsonify({'error': 'Unsupported Content-Type'}), 400

    if not N or not P or not K:
        return jsonify({'error': 'Missing required parameter(s)'}), 400
    print(f'The values are {N},{P},{K}')
    try:
        # create an instance of CropPredictor class
        crop_predictor = CropPredictor(N, P, K)
        crop = crop_predictor.predict()

        crop_dict = {'crop': str(crop.tolist()[0])}

        return jsonify(crop_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)