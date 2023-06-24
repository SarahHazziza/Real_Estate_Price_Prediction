from flask import Flask, render_template, request
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model


app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['Area'])
    city = request.form['City']
    city_area = request.form['city_area']
    p_type = request.form['type']
    has_elevator = int(request.form['hasElevator '])
    has_storage = int(request.form['hasStorage '])
    has_air_condition = int(request.form['hasAirCondition '])
    has_balcony = int(request.form['hasBalcony '])
    entrance_date = request.form['entranceDate ']
    condition = request.form['condition ']
    furniture = request.form['furniture ']
    


    Test = pd.DataFrame({
    'Area': [area],
    'City': [city],
    'city_area': [city_area],
    'type': [p_type],
    'hasElevator ': [has_elevator],
    'hasStorage ': [has_storage],
    'hasAirCondition ': [has_air_condition],
    'hasBalcony ': [has_balcony],
    'entranceDate ': [entrance_date],
    'condition ': [condition],
    'furniture ': [furniture],
    'floor' : [np.NaN],
    'num_of_images' : [np.NaN],
    'Street' : [np.NaN]
})

    prediction = model.predict(Test)

    return render_template('index.html', prediction=np.round(prediction,2))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=True)
