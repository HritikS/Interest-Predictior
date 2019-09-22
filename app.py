from flask import Flask, render_template, request
import urllib3.request
import requests
from VH import predict
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def form():
    X_pred = []
    for i in range(1, 16):
        X_pred.append(int(request.form['r' + str(i)]))
    categ = ['Public Speaking', 'Art', 'Programming', 'Dramatics', 'Sports']
    X_pred = np.array(X_pred)
    X_pred = X_pred.reshape(1,-1)
    cat = predict(X_pred)
    b = []
    for i in range(5):
        if(cat[0][i] >= 0.4):
            b.append(categ[i])
    return render_template('index.html', ans=b)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port='5000', debug=True)