from flask import Flask, render_template, request, redirect, url_for
from predict import predict
import os
# import cv2
# import matplotlib.pyplot as plt

app = Flask('image_description')

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')

@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        myfile = request.files['myfile']
        predicted_desc = predict(myfile).replace('start ','').replace('end','')
        del myfile
        return predicted_desc


@app.route('/refresh')
def refresh():
    return render_template('body.html')


if __name__ == "__main__":
    app.run("localhost", "9999", debug=True, threaded=False)
