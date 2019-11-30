from flask import Flask, request, render_template, jsonify
from predict import *
import subprocess
import io
from flask_cors import CORS
import urllib.request

app = Flask(__name__)
CORS(app)

def str_l_to_l(s):
    return s.strip('][').split(', ')

@app.route('/predict_api', methods=['GET', 'POST'])
def predict_classes():
    if request.method == 'GET':
        return render_template('home.html', value = "Image")
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Image not uploaded"
        file = request.files['file'].read()
        # req = request.get_json()
        # print("REQ: {}, Type: {}".format(req, type(req)))
        # file = request.form['file']
        # file = req
        try:
            img = Image.open(io.BytesIO(file))
            # img = Image.open(urllib.request.urlopen(file))
        except IOError:
            return jsonify(predictions = "Not an Image, Upload a proper image file", preds_prob = "")
        # img = Image.open(io.BytesIO(file))
        img = img.convert("RGB")
        img.save("input.jpg", "JPEG")
        p = (subprocess.check_output(["python /Users/vatsalsaglani/Desktop/njunk/personal/CelebA_API/predict.py input.jpg"], shell=True).decode("utf-8"))
        pp = p.split("@")
        preds = pp[0]
        pred_proba = pp[1]
        pred_proba = pred_proba.split("-")

        
        
        return jsonify(predicitions = preds, preds_prob = pred_proba)

if __name__ == '__main__':
    app.run(debug = True)