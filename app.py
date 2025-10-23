import pickle
import json
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regression_model = pickle.load(open('regression_model.pkl', 'rb'))
scalar_model = pickle.load(open('scaler_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar_model.transform(np.array(list(data.values())).reshape(1,-1))
    output = regression_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar_model.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regression_model.predict(final_input)[0]
    return render_template("home.html", prediction_text="The house price prediction is: {}".format(output))

if __name__=='__main__':
    app.run(debug=True)