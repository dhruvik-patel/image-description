from flask import Flask, render_template, request, redirect, url_for
from predict import predict
import os

app = Flask('image_description')

@app.route('/')
def show_predict_stock_form():
    return render_template('home.html')

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
