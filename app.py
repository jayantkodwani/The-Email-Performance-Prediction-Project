import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('email_open_model.sav','rb'))
model_click = pickle.load(open('email_click_model.sav','rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    
    #int_features = [float(x) for x in request.form.values()]
    
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    
    int_features = request.form.get('Sent')

    playSel = request.form.get('play')
    playSel_num = (playSel.split('.')[0])
   
    solutionareaSel = request.form.get('solutionarea')
    solutionarea_num = (solutionareaSel.split('.')[0])

    final_features = [np.array([int_features,playSel_num,solutionarea_num])]
    prediction = model.predict(final_features)
    output = round(prediction[0])
    #print(prediction[0])

    final_features_click = [np.array([output,playSel_num,solutionarea_num])]
    prediction2 = model_click.predict(final_features_click)    
    return render_template('home.html', prediction_text="Estimated number of opens: {}".format(round(prediction[0]),0),prediction_text_click="Estimated number of Clicks: {}".format(round(prediction2[0]),0))
    #return render_template('home.html', prediction_text="Estimated number of opens: {}".format(round(prediction[0]),0))
    #return render_template('home.html', prediction_text_click="Estimated number of Clicks: {}".format(round(prediction2[0]),0))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)