#!/usr/bin/env python3
from predict import mnist
from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import base64
import re

app = Flask(__name__)
predictor = mnist.NumberPredictor(60000, 2)

@app.route('/')
def render_home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    r = request
    data = r.get_data()    
    
    decoded = Image.open(BytesIO(base64.b64decode(data.split(b',')[1]))).convert("L")
    decoded = decoded.resize((28,28))
    nparr = np.array(decoded)
    nparr = 255 - nparr
    
    return str(predictor.predict(nparr.flatten()))

if __name__ == "__main__":
    app.run(host="localhost")
