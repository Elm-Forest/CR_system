import os

import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from flask_cors import cross_origin, CORS

from exe import Cloud_Removal
from web_service.app.service import preprocess_service, infer_service, upload_service

graph = tf.Graph()
with graph.as_default():
    session = tf.Session(graph=graph)
    with session.as_default():
        cr_exe = Cloud_Removal()

app = Flask(__name__)
app.static_folder = 'static'

CORS(app)


@app.route('/')
def index():
    return render_template('app.html')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    if 'cloudy_image' in request.files and 'sar_image' in request.files and 'target_image' in request.files:
        cloudy_image = request.files['cloudy_image']
        sar_image = request.files['sar_image']
        target_image = request.files['target_image']
        cloudy_image_name = upload_service(cloudy_image, sar_image, target_image)
        return {'image_name': os.path.splitext(cloudy_image_name)[0]}
    return "Files not found", 400


@app.route('/preprocess', methods=['POST'])
@cross_origin()
def preprocess():
    data = request.json
    image_name = data['image_name']
    result = preprocess_service(image_name, cr_exe)
    return jsonify(result)


@app.route('/infer', methods=['POST'])
@cross_origin()
def infer():
    data = request.json
    image_name = data['image_name']
    with graph.as_default():
        with session.as_default():
            result = infer_service(image_name, cr_exe)
    return jsonify(result)


if __name__ == '__main__':
    port = 7890
    app.run(port=port)
