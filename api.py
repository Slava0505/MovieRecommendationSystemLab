#!/usr/bin/env python3

import os

from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename
from threading import Lock
from src.model import BaseRecommendationModel

app = Flask(__name__)
app.secret_key = os.urandom(128)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'txt'}
LOCK = Lock()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def process():
    if 'file' not in request.files\
        or request.files['file'].filename == ''\
        or not allowed_file(request.files['file'].filename):
        return make_response(jsonify({'error': 'No file, or file has invalid extension!'}), 403)

    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)
    
    with LOCK:
        query = request.files['file']
        filename = secure_filename(query.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        query.save(filepath)
        try:
            result = predict_categories(filepath)
        except Exception as e:
            print(e)
            return make_response(jsonify({'error': 'Some error during classification!'}), 500)
        else:
            return make_response(jsonify({'result': '/3484eb2a-46be-4b55-b832-a260995480ee/' + result}), 200)

@app.route('/train', methods=['POST'])
def process():
    if 'file' not in request.files\
        or request.files['file'].filename == ''\
        or not allowed_file(request.files['file'].filename):
        return make_response(jsonify({'error': 'No file, or file has invalid extension!'}), 403)

    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)
    
    with LOCK:
        query = request.files['file']
        filename = secure_filename(query.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        query.save(filepath)
        try:
            result = predict_categories(filepath)
        except Exception as e:
            print(e)
            return make_response(jsonify({'error': 'Some error during classification!'}), 500)
        else:
            return make_response(jsonify({'result': '/3484eb2a-46be-4b55-b832-a260995480ee/' + result}), 200)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)