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
def predict():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar_model.transform(np.array(list(data.values())).reshape(1,-1))
    output = regression_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__=='__main__':
    app.run(debug=True)