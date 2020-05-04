import numpy as np
from flask import Flask, abort, jsonify, request
import pickle

model = pickle.load(open("credit_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def prediction():
    data = request.get_json(force=True)
    print(data)
    #features = list(data[0].keys())
    features = list(data.keys())
    predict_request = []
    for feature in features:
        predict_request.insert(len(predict_request), data[feature])

    predict_request = np.array(predict_request).reshape(1, -1)
    y_hat = model.predict(predict_request)
    output = {'y_hat': int(y_hat[0])}
    print(output)
    return 'jsonify(output)'
    
    #print(data[0]["LIMIT_BAL"])
    #transforms go here
    #data = {"LIMIT RANGE":"1","SEX":2,"EDUCATION":2,"MARRIAGE":2,"AGE RANGE":"1","PAY_0":-1,"PAY_2":2,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":2,"BILL_AMT1":2682,"BILL_AMT2":1725,"BILL_AMT3":2682,"BILL_AMT4":3272,"BILL_AMT5":3455,"BILL_AMT6":3261,"PAY_AMT1":0,"PAY_AMT2":1000,"PAY_AMT3":1000,"PAY_AMT4":1000,"PAY_AMT5":0,"PAY_AMT6":2000}
    #predict_request = [data] #33min
    #reshape data to go into model
    #predict_request = np.array(data).reshape(1, -1)
    #print(predict_request)
    #y_hat = model.predict(data)
    #print(y_hat)
    #ouput = {'y_hat': y_hat[0]}
    #return jsonify(response=output)
    return 'hello'
if __name__ == '__main__':
    app.run(port=9000, debug=True)